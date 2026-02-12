use crate::data::Data;
use crate::data::DataSource;
use crate::index::{Match, Search, SearchParamsExt};
use crate::utils::{load_u32_vec, load_usize_vec_from_u64, progress_bar};
use anyhow::{Result, anyhow};
use itertools::Itertools;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use pathfinding::{kuhn_munkres::kuhn_munkres, matrix::Matrix};
use serde::Deserialize;
use serde_aux::prelude::*;
use std::fs::create_dir_all;
use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeSet, HashMap, HashSet, hash_map::Entry},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

/// Normalize text for indexing and search
pub fn normalize(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }
    let norm = any_ascii::any_ascii(s);
    let norm = if norm.is_empty() {
        // if ascii conversion produces an empty string, return lowercase as fallback
        s.to_lowercase()
    } else {
        norm.to_lowercase()
    };
    // remove all punctuation characters around words
    // but keep punctuation inside words and words containing only punctuation
    norm.split_whitespace()
        .map(|word| {
            let trimmed = word
                .trim_end_matches(|c: char| c.is_ascii_punctuation())
                .trim_start_matches(|c: char| c.is_ascii_punctuation());
            // only punctuation
            if trimmed.is_empty() {
                word.to_string()
            } else {
                trimmed.to_string()
            }
        })
        .join(" ")
}

fn lower_bound<F>(mut start: usize, mut end: usize, cmp: F) -> Option<(usize, bool)>
where
    F: Fn(usize) -> Ordering,
{
    let mut answer = None;

    while start < end {
        let mid = (start + end) / 2;
        match cmp(mid) {
            Ordering::Less => {
                start = mid + 1;
            }
            ord @ (Ordering::Equal | Ordering::Greater) => {
                answer = Some((mid, ord == Ordering::Equal));
                end = mid;
            }
        }
    }
    answer
}

fn upper_bound<F>(mut start: usize, mut end: usize, cmp: F) -> Option<usize>
where
    F: Fn(usize) -> Ordering,
{
    let mut answer = None;

    while start < end {
        let mid = (start + end) / 2;
        match cmp(mid) {
            Ordering::Less | Ordering::Equal => {
                start = mid + 1;
            }
            Ordering::Greater => {
                answer = Some(mid);
                end = mid;
            }
        }
    }
    answer
}

struct InvList<'a> {
    word: usize,
    exact: bool,
    length: usize,
    inv_list: &'a [u8],
}

impl InvList<'_> {
    fn parse(&self) -> Result<&[u32]> {
        let (head, body, tail) = unsafe { self.inv_list.align_to::<u32>() };
        if !head.is_empty() || !tail.is_empty() {
            return Err(anyhow!("Inverted list not aligned"));
        }
        Ok(body)
    }
}

#[derive(Debug)]
struct Item {
    candidate: Option<Candidate>,
    matches: HashMap<(usize, usize), Vec<(usize, f32)>>,
}

impl Item {
    const ES: f32 = 1.0; // exact match score
    const PS: f32 = 0.75; // prefix match score
    const KP: f32 = 0.5; // keyword penalty
    const WP: f32 = 0.25; // word penalty

    fn new() -> Self {
        Self {
            candidate: None,
            matches: HashMap::new(),
        }
    }

    fn update(&mut self, word: usize, occurrence: usize, keyword: usize, exact: bool) {
        self.matches
            .entry((word, occurrence))
            .or_default()
            .push((keyword, if exact { Self::ES } else { Self::PS }));
    }

    fn candidate(
        &self,
        field_id: u32,
        data_id: u32,
        num_keywords: usize,
        length: u32,
    ) -> Candidate {
        let score = self.assignment(num_keywords);

        let num_matches = num_keywords.min(self.matches.len());
        let unmatched_keywords = num_keywords.saturating_sub(num_matches);
        // make sure to not penalize twice for unmatched keyword and word
        let unmatched_words = (length as usize)
            .saturating_sub(num_matches)
            .saturating_sub(unmatched_keywords);

        let penalty = unmatched_keywords as f32 * Self::KP + unmatched_words as f32 * Self::WP;

        let score = score - penalty;
        Candidate::new(field_id, score, data_id)
    }

    fn assignment(&self, num_keywords: usize) -> f32 {
        // make sure to have at least as many columns as rows
        let num_columns = num_keywords.max(self.matches.len());
        let weights = Matrix::from_rows(self.matches.values().map(|matches| {
            let mut row = vec![OrderedFloat(0.0); num_columns];
            for (keyword, score) in matches {
                row[*keyword] = OrderedFloat(*score);
            }
            row
        }))
        .expect("Cannot fail");

        let (score, _) = kuhn_munkres(&weights);
        score.into_inner()
    }
}

#[derive(Debug, Copy, Clone)]
struct Candidate {
    score: OrderedFloat<f32>,
    data_id: u32,
    field_id: u32,
}

impl Candidate {
    fn new(field_id: u32, score: f32, data_id: u32) -> Self {
        Self {
            score: OrderedFloat(score),
            data_id,
            field_id,
        }
    }

    fn add(&self, score: f32) -> Self {
        Self {
            score: self.score + score,
            data_id: self.data_id,
            field_id: self.field_id,
        }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
            && self.data_id == other.data_id
            && self.field_id == other.field_id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.data_id.cmp(&self.data_id))
            .then_with(|| self.field_id.cmp(&other.field_id))
    }
}

const U32_SIZE: usize = size_of::<u32>();

#[derive(Debug)]
struct Inner {
    data: Data,
    keywords: Mmap,
    keyword_offsets: Vec<usize>,
    inv_lists: Mmap,
    inv_list_offsets: Vec<usize>,
    lengths: Vec<u32>,
    field_to_data: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct KeywordIndex {
    inner: Arc<Inner>,
}

impl KeywordIndex {
    #[inline]
    fn prefix_cmp(word: &str, prefix: &str) -> Ordering {
        // prefix comparison
        // 1. return equal if prefix is prefix of word or equal
        // 2. return less if word is less than prefix
        // 3. return greater if word is greater than prefix
        let mut wi = 0;
        let mut pi = 0;

        let word = word.as_bytes();
        let prefix = prefix.as_bytes();

        while wi < word.len() && pi < prefix.len() {
            match word[wi].cmp(&prefix[pi]) {
                Ordering::Equal => {
                    wi += 1;
                    pi += 1;
                }
                ordering => return ordering,
            }
        }
        if pi == prefix.len() {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }

    #[inline]
    fn get_inverted_list(&self, word: usize) -> (&[u8], usize) {
        let inner = self.inner.as_ref();
        let start = inner.inv_list_offsets[word];
        let end = inner
            .inv_list_offsets
            .get(word + 1)
            .copied()
            .unwrap_or_else(|| inner.inv_lists.len());

        let data = &inner.inv_lists[start..end];
        let length = data.len() / U32_SIZE;
        (data, length)
    }

    #[inline]
    fn get_word(&self, word: usize) -> &str {
        let inner = self.inner.as_ref();
        let start = inner.keyword_offsets[word];
        let end = inner
            .keyword_offsets
            .get(word + 1)
            .copied()
            .unwrap_or_else(|| inner.keywords.len());
        unsafe { std::str::from_utf8_unchecked(&inner.keywords[start..end]) }
    }

    fn field_to_column(&self, field_id: u32) -> usize {
        let field_to_data = &self.inner.field_to_data;

        let mut field_id = field_id as usize;
        let data_id = field_to_data[field_id] as usize;

        let mut offset = 0;
        while field_id > 0 && field_to_data[field_id - 1] == data_id as u32 {
            field_id -= 1;
            offset += 1;
        }
        offset
    }

    fn num_keywords(&self) -> usize {
        self.inner.keyword_offsets.len()
    }

    fn get_matches(&self, prefix: &str) -> Vec<InvList<'_>> {
        let mut matches = vec![];

        let lower = match lower_bound(0, self.num_keywords(), |word| {
            self.get_word(word).cmp(prefix)
        }) {
            None => return matches,
            Some((word, true)) => {
                let (inv_list, length) = self.get_inverted_list(word);
                matches.push(InvList {
                    word,
                    exact: true,
                    length,
                    inv_list,
                });
                word.saturating_add(1)
            }
            Some((word, false)) => word,
        };

        let upper = upper_bound(lower, self.num_keywords(), |word| {
            Self::prefix_cmp(self.get_word(word), prefix)
        })
        .unwrap_or_else(|| self.num_keywords());

        matches.extend((lower..upper).map(|word| {
            let (inv_list, length) = self.get_inverted_list(word);
            InvList {
                word,
                exact: false,
                length,
                inv_list,
            }
        }));

        matches
    }

    fn search_internal<F>(
        &self,
        query: &str,
        params: &KeywordSearchParams,
        filter: Option<F>,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        let data = &self.inner.data;
        let field_to_data = &self.inner.field_to_data;
        let lengths = &self.inner.lengths;

        let search_k = params.search_k(data);

        let filter = move |id| {
            if let Some(ref filter) = filter {
                filter(id)
            } else {
                // keep everything if no filter provided
                true
            }
        };

        let query = normalize(query);
        let mut keywords: Vec<_> = query.split_whitespace().collect();
        let num_keywords = keywords.len();

        let mut items: HashMap<u32, Item> = HashMap::new();

        let keyword_matches: HashMap<_, _> = keywords
            .iter()
            .unique()
            .map(|&kw| {
                let inv_lists = self.get_matches(kw);
                let total_length: usize = inv_lists.iter().map(|inv_list| inv_list.length).sum();
                (kw, (inv_lists, total_length))
            })
            .collect();

        keywords.sort_by_key(|kw| keyword_matches[kw].1);

        let mut top_k: BTreeSet<Candidate> = BTreeSet::new();

        let worst_candidate = |top_k: &BTreeSet<_>| -> Option<Candidate> {
            if top_k.len() == search_k {
                top_k.first().copied()
            } else if top_k.len() > search_k {
                panic!("top_k has more than k elements, should not happen");
            } else {
                None
            }
        };

        let mut skip = HashSet::new();

        for (keyword, kw) in keywords.into_iter().enumerate() {
            let (inv_lists, _) = &keyword_matches[kw];
            let keywords_left = num_keywords - keyword - 1;
            let max_future_score = Item::ES * keywords_left as f32;

            for inv_list in inv_lists {
                let mut last_field_id = None;
                let mut occurrence = 0;
                for &field_id in inv_list.parse()? {
                    if skip.contains(&field_id) {
                        continue;
                    }

                    let data_id = field_to_data[field_id as usize];
                    if !filter(data_id) {
                        continue;
                    }

                    if Some(field_id) == last_field_id {
                        occurrence += 1;
                    } else {
                        occurrence = 0;
                    }
                    last_field_id = Some(field_id);

                    let length = lengths[field_id as usize];

                    // update item and get current score
                    let mut entry = items.entry(field_id);
                    let item = match entry {
                        Entry::Occupied(ref mut entry) => entry.get_mut(),
                        Entry::Vacant(entry) => {
                            if let Some(worst) = worst_candidate(&top_k) {
                                // compute an upper bound score for this newly matched field_id
                                let mut upper_bound_score = max_future_score;
                                upper_bound_score +=
                                    if inv_list.exact { Item::ES } else { Item::PS };
                                upper_bound_score -= keyword as f32 * Item::KP;

                                let upper_bound =
                                    Candidate::new(field_id, upper_bound_score, data_id);
                                if upper_bound <= worst {
                                    // even the upper bound is not enough to enter the top k
                                    skip.insert(field_id);
                                    continue;
                                }
                            }
                            entry.insert(Item::new())
                        }
                    };
                    item.update(inv_list.word, occurrence, keyword, inv_list.exact);
                    let current = item.candidate(field_id, data_id, num_keywords, length);
                    let old = item.candidate.replace(current);
                    if let Some(old) = old {
                        // remove old candidate from top_k, might not be present
                        top_k.remove(&old);
                    }

                    let Some(worst) = worst_candidate(&top_k) else {
                        // top_k not full yet, just insert
                        top_k.insert(current);
                        continue;
                    };

                    // upper bound for current candidate if all future keywords match exactly
                    let upper_bound = current.add(max_future_score);
                    if upper_bound <= worst {
                        // even in the best case this item cannot enter the top k
                        skip.insert(field_id);
                        continue;
                    }

                    if current > worst {
                        // better than the worst in top_k, insert
                        top_k.pop_first();
                        top_k.insert(current);
                    }
                }
            }
        }

        let matches: Vec<_> = top_k
            .into_iter()
            .sorted_by_key(|&candidate| {
                (
                    candidate.data_id,
                    Reverse(candidate.score),
                    candidate.field_id,
                )
            })
            .dedup_by(|a, b| a.data_id == b.data_id)
            .sorted_by_key(|&candidate| (Reverse(candidate.score), candidate.data_id))
            .filter_map(|candidate| {
                let score = candidate.score.into_inner();
                if let Some(min_score) = params.min_score
                    && score < min_score
                {
                    return None;
                }
                Some(Match::WithField(
                    candidate.data_id,
                    self.field_to_column(candidate.field_id),
                    score,
                ))
            })
            .take(params.k)
            .collect();

        Ok(matches)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct KeywordSearchParams {
    #[serde(
        default = "crate::index::default_k",
        deserialize_with = "deserialize_number_from_string"
    )]
    pub k: usize,
    #[serde(
        default,
        rename = "min-score",
        deserialize_with = "deserialize_option_number_from_string"
    )]
    pub min_score: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_bool_from_anything")]
    pub exact: bool,
}

impl Default for KeywordSearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            min_score: None,
            exact: false,
        }
    }
}

impl SearchParamsExt for KeywordSearchParams {
    fn k(&self) -> usize {
        self.k
    }

    fn exact(&self) -> bool {
        self.exact
    }
}

impl Search for KeywordIndex {
    type Data = Data;
    type Query<'q> = &'q str;
    type BuildParams = ();
    type SearchParams = KeywordSearchParams;

    fn build(data: &Self::Data, index_dir: &Path, _params: &Self::BuildParams) -> Result<()> {
        create_dir_all(index_dir)?;

        let mut inv_lists: HashMap<String, Vec<u32>> = HashMap::new();

        let mut field_to_data_file =
            BufWriter::new(File::create(index_dir.join("index.field-to-data"))?);
        let mut lengths_file = BufWriter::new(File::create(index_dir.join("index.lengths"))?);

        let total_fields = data.total_fields();
        let pb = progress_bar("Building keyword index", Some(total_fields as u64))?;

        let mut field_id = 0;
        for (id, fields) in data.items() {
            for field in fields.into_iter() {
                if field_id == u32::MAX {
                    return Err(anyhow!("too many fields, max {} supported", u32::MAX));
                }

                if !field.is_text() {
                    return Err(anyhow!("Keyword index can only be built on text fields"));
                }

                let text = normalize(field.as_str());

                let mut length: u32 = 0;
                for word in text.split_whitespace() {
                    let inv_list = inv_lists.entry(word.to_string()).or_default();
                    inv_list.push(field_id);
                    length += 1;
                }
                field_id += 1;
                field_to_data_file.write_all(&id.to_le_bytes())?;
                lengths_file.write_all(&length.to_le_bytes())?;

                pb.inc(1);
            }
        }

        pb.finish_with_message("Keyword index built");

        // first sort by key to have them in lexicographical order
        let mut keyword_file = BufWriter::new(File::create(index_dir.join("index.keywords"))?);
        let mut inv_list_file = BufWriter::new(File::create(index_dir.join("index.inv-lists"))?);
        let mut keyword_offset_file =
            BufWriter::new(File::create(index_dir.join("index.keyword-offsets"))?);
        let mut inv_list_offset_file =
            BufWriter::new(File::create(index_dir.join("index.inv-list-offsets"))?);
        let mut keyword_offset = 0;
        let mut inv_list_offset = 0;

        for (keyword, inv_list) in inv_lists.into_iter().sorted_by(|(a, _), (b, _)| a.cmp(b)) {
            // write keyword and offset
            keyword_file.write_all(keyword.as_bytes())?;
            let keyword_offset_bytes = u64::try_from(keyword_offset)?.to_le_bytes();
            keyword_offset_file.write_all(&keyword_offset_bytes)?;
            keyword_offset += keyword.len();

            // write inverted list offset
            let inv_list_offset_bytes = u64::try_from(inv_list_offset)?.to_le_bytes();
            inv_list_offset_file.write_all(&inv_list_offset_bytes)?;

            // write inverted list
            for field_id in &inv_list {
                inv_list_file.write_all(&field_id.to_le_bytes())?;
            }
            inv_list_offset += inv_list.len() * U32_SIZE;
        }
        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self> {
        let keywords = unsafe { Mmap::map(&File::open(index_dir.join("index.keywords"))?)? };
        let inv_lists = unsafe { Mmap::map(&File::open(index_dir.join("index.inv-lists"))?)? };

        let keyword_offsets = load_usize_vec_from_u64(&index_dir.join("index.keyword-offsets"))?;
        let inv_list_offsets = load_usize_vec_from_u64(&index_dir.join("index.inv-list-offsets"))?;

        let lengths = load_u32_vec(&index_dir.join("index.lengths"))?;
        let field_to_data = load_u32_vec(&index_dir.join("index.field-to-data"))?;

        let inner = Arc::new(Inner {
            data,
            keywords,
            keyword_offsets,
            inv_lists,
            inv_list_offsets,
            field_to_data,
            lengths,
        });

        Ok(Self { inner })
    }

    fn data(&self) -> &Self::Data {
        &self.inner.data
    }

    fn index_type(&self) -> &'static str {
        "keyword"
    }

    fn search(&self, query: Self::Query<'_>, params: &Self::SearchParams) -> Result<Vec<Match>> {
        self.search_internal(query, params, None::<fn(u32) -> bool>)
    }

    fn search_with_filter<F>(
        &self,
        query: Self::Query<'_>,
        params: &Self::SearchParams,
        filter: F,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        self.search_internal(query, params, Some(filter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        Data,
        item::{FieldType, Item as DataItem, StringField},
    };
    use std::fs::create_dir_all;
    use tempfile::tempdir;

    fn build_keyword_index(data: Data, index_dir: &Path) -> KeywordIndex {
        create_dir_all(index_dir).expect("Failed to create index directory");
        KeywordIndex::build(&data, index_dir, &()).expect("Failed to build index");
        KeywordIndex::load(data, index_dir).expect("Failed to load index")
    }

    #[test]
    fn test_special_keyword_index() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        // Create test data: id -> labels
        let items = vec![
            Ok(DataItem::from_string_fields(
                "0".to_string(),
                vec![StringField {
                    field_type: FieldType::Text,
                    value: "agar".to_string(),
                    tags: vec![],
                }],
            )
            .expect("Failed to create Item")),
            Ok(DataItem::from_string_fields(
                "1".to_string(),
                vec![StringField {
                    field_type: FieldType::Text,
                    value: "agar agar".to_string(),
                    tags: vec![],
                }],
            )
            .expect("Failed to create Item")),
        ];

        Data::build(items, &data_dir).expect("Failed to build data");
        let data = Data::load(&data_dir).expect("Failed to load data");

        let index = build_keyword_index(data, &index_dir);

        let matches = index
            .search(
                "agar",
                &KeywordSearchParams {
                    k: 2,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");
        assert_eq!(matches.len(), 2);
        assert!(matches!(matches[0], Match::WithField(0, 0, score) if (score - 1.0).abs() < 1e-6));
        assert!(matches!(matches[1], Match::WithField(1, 0, score) if (score - 0.75).abs() < 1e-6));

        let matches = index
            .search(
                "agar agar",
                &KeywordSearchParams {
                    k: 2,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");
        assert_eq!(matches.len(), 2);
        assert!(matches!(matches[0], Match::WithField(1, 0, score) if (score - 2.0).abs() < 1e-6));
        assert!(matches!(matches[1], Match::WithField(0, 0, score) if (score - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_keyword_index() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        // Create simple test data
        let items = vec![Ok(DataItem::from_string_fields(
            "Q30".to_string(),
            vec![
                StringField {
                    field_type: FieldType::Text,
                    value: "United States".to_string(),
                    tags: vec![],
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "the U.S. of A".to_string(),
                    tags: vec![],
                },
            ],
        )
        .expect("Failed to create Item"))];

        Data::build(items, &data_dir).expect("Failed to build data");
        let data = Data::load(&data_dir).expect("Failed to load data");

        let index = build_keyword_index(data, &index_dir);

        // Test exact match
        let matches = index
            .search(
                "United States",
                &KeywordSearchParams {
                    k: 1,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");

        assert!(matches!(matches[0], Match::WithField(0, 0, score) if (score - 2.0).abs() < 1e-6));

        // Test partial match
        let matches = index
            .search(
                "United State",
                &KeywordSearchParams {
                    k: 1,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");

        assert!(matches!(matches[0], Match::WithField(0, 0, score) if (score - 1.75).abs() < 1e-6));

        // Test synonym match
        let matches = index
            .search(
                "the U.S. of A",
                &KeywordSearchParams {
                    k: 1,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");

        assert!(matches!(matches[0], Match::WithField(0, 1, score) if (score - 4.0).abs() < 1e-6));
        assert_eq!(index.data().field(0, 1).unwrap().as_str(), "the U.S. of A");

        // Test no match
        let matches = index
            .search(
                "theunitedstates",
                &KeywordSearchParams {
                    k: 1,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");

        assert!(matches.is_empty());
    }

    #[test]
    fn test_normalize() {
        // empty string
        assert_eq!(normalize(""), "");

        // basic lowercase conversion
        assert_eq!(normalize("Hello World"), "hello world");

        // remove punctuation at word boundaries
        assert_eq!(normalize("Hello, World!"), "hello world");
        assert_eq!(normalize("(Hello) [World]"), "hello world");

        // keep punctuation inside words
        assert_eq!(normalize("it's a test"), "it's a test");
        assert_eq!(normalize("semi-automated"), "semi-automated");

        // handle words with only punctuation
        assert_eq!(normalize("Hello --- World"), "hello --- world");

        // handle non-ASCII characters
        assert_eq!(normalize("Café"), "cafe");
        assert_eq!(normalize("Größe"), "grosse");
        assert_eq!(normalize("Niño"), "nino");

        // handle emojis
        assert_eq!(normalize("Hello 😊 World"), "hello blush world");

        // combination of cases
        assert_eq!(normalize("!Hello, Größe-Test!"), "hello grosse-test");
    }

    #[test]
    fn test_bounds() {
        let values = [1, 1, 2, 3, 3, 5, 8, 8, 9];
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&3)),
            Some((3, true))
        );
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&4)),
            Some((5, false))
        );
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&0)),
            Some((0, false))
        );
        assert_eq!(lower_bound(0, values.len(), |i| values[i].cmp(&10)), None);
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&8)),
            Some((6, true))
        );

        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&3)), Some(5));
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&4)), Some(5));
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&10)), None);
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&8)), Some(8));

        let values: Vec<i32> = vec![];
        assert_eq!(lower_bound(0, values.len(), |i| values[i].cmp(&3)), None);
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&3)), None);

        let values = [2, 2];
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&2)),
            Some((0, true))
        );
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&2)), None);

        let values = [1, 2, 2, 4];
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&2)),
            Some((1, true))
        );
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&2)), Some(3));
    }

    #[test]
    fn test_field_matching() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        // Create test data with multiple fields per entity
        // Entity 0: ["common label", "specific alpha", "specific beta"]
        // Entity 1: ["common label", "specific gamma", "specific delta"]
        let items = vec![
            Ok(DataItem::from_string_fields(
                "Entity0".to_string(),
                vec![
                    StringField {
                        field_type: FieldType::Text,
                        value: "common label".to_string(),
                        tags: vec![],
                    },
                    StringField {
                        field_type: FieldType::Text,
                        value: "specific alpha".to_string(),
                        tags: vec![],
                    },
                    StringField {
                        field_type: FieldType::Text,
                        value: "specific beta".to_string(),
                        tags: vec![],
                    },
                ],
            )
            .expect("Failed to create Item")),
            Ok(DataItem::from_string_fields(
                "Entity1".to_string(),
                vec![
                    StringField {
                        field_type: FieldType::Text,
                        value: "common label".to_string(),
                        tags: vec![],
                    },
                    StringField {
                        field_type: FieldType::Text,
                        value: "specific gamma".to_string(),
                        tags: vec![],
                    },
                    StringField {
                        field_type: FieldType::Text,
                        value: "specific delta".to_string(),
                        tags: vec![],
                    },
                ],
            )
            .expect("Failed to create Item")),
        ];

        Data::build(items, &data_dir).expect("Failed to build data");
        let data = Data::load(&data_dir).expect("Failed to load data");
        let index = build_keyword_index(data, &index_dir);

        // Test 1: Query "specific alpha" should match Entity0, field 1
        let matches = index
            .search(
                "specific alpha",
                &KeywordSearchParams {
                    k: 10,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to search");

        assert_eq!(matches.len(), 2);
        if let Match::WithField(id, field_idx, score) = matches[0] {
            assert_eq!(id, 0, "Expected Entity0");
            assert_eq!(field_idx, 1, "Expected field 1 (not the first field)");
            assert!(score > 1.0, "Expected high score for exact match");

            // Verify actual field text
            let field_text = index
                .data()
                .field(id, field_idx)
                .expect("Field should exist");
            assert_eq!(field_text.as_str(), "specific alpha");
        } else {
            panic!("Expected Match::WithField");
        }

        // Test 2: Query "specific beta" should match Entity0, field 2
        let matches = index
            .search(
                "specific beta",
                &KeywordSearchParams {
                    k: 10,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to search");

        assert_eq!(matches.len(), 2);
        if let Match::WithField(id, field_idx, score) = matches[0] {
            assert_eq!(id, 0, "Expected Entity0");
            assert_eq!(field_idx, 2, "Expected field 2");
            assert!(score > 1.0, "Expected high score for exact match");

            let field_text = index
                .data()
                .field(id, field_idx)
                .expect("Field should exist");
            assert_eq!(field_text.as_str(), "specific beta");
        } else {
            panic!("Expected Match::WithField");
        }

        // Test 3: Query "specific gamma" should match Entity1, field 1
        let matches = index
            .search(
                "specific gamma",
                &KeywordSearchParams {
                    k: 10,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to search");

        assert_eq!(matches.len(), 2);
        if let Match::WithField(id, field_idx, score) = matches[0] {
            assert_eq!(id, 1, "Expected Entity1");
            assert_eq!(field_idx, 1, "Expected field 1");
            assert!(score > 1.0, "Expected high score for exact match");

            let field_text = index
                .data()
                .field(id, field_idx)
                .expect("Field should exist");
            assert_eq!(field_text.as_str(), "specific gamma");
        } else {
            panic!("Expected Match::WithField");
        }

        // Test 4: Query "specific delta" should match Entity1, field 2
        let matches = index
            .search(
                "specific delta",
                &KeywordSearchParams {
                    k: 10,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to search");

        assert_eq!(matches.len(), 2);
        if let Match::WithField(id, field_idx, score) = matches[0] {
            assert_eq!(id, 1, "Expected Entity1");
            assert_eq!(field_idx, 2, "Expected field 2");
            assert!(score > 1.0, "Expected high score for exact match");

            let field_text = index
                .data()
                .field(id, field_idx)
                .expect("Field should exist");
            assert_eq!(field_text.as_str(), "specific delta");
        } else {
            panic!("Expected Match::WithField");
        }

        // Test 5: Query "common" should match both entities but return field 0 for each
        let matches = index
            .search(
                "common",
                &KeywordSearchParams {
                    k: 10,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to search");

        assert_eq!(matches.len(), 2);
        for m in matches {
            if let Match::WithField(id, field_idx, score) = m {
                assert_eq!(field_idx, 0, "Expected field 0 for 'common' query");
                assert!(score > 0.0, "Expected positive score");

                let field_text = index
                    .data()
                    .field(id, field_idx)
                    .expect("Field should exist");
                assert_eq!(field_text.as_str(), "common label");
            } else {
                panic!("Expected Match::WithField");
            }
        }
    }
}
