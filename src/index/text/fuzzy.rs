use crate::data::Data;
use crate::data::DataSource;
use crate::index::{Match, Search, SearchParamsExt};
use crate::utils::{load_u32_vec, load_usize_vec_from_u64, progress_bar};
use anyhow::{Result, anyhow};
use fst::automaton::Levenshtein;
use fst::{IntoStreamer, Map, Streamer};
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
    io::{BufWriter, Write as _},
    path::Path,
    sync::Arc,
};

use super::keyword::normalize;

const U32_SIZE: usize = size_of::<u32>();

struct InvList<'a> {
    word: usize,
    score: f32,
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
    const F1: f32 = 0.75; // fuzzy distance 1 score
    const F2: f32 = 0.5; // fuzzy distance 2 score
    const F3: f32 = 0.25; // fuzzy distance 3+ score
    const KP: f32 = 0.5; // keyword penalty
    const WP: f32 = 0.25; // word penalty

    fn new() -> Self {
        Self {
            candidate: None,
            matches: HashMap::new(),
        }
    }

    fn update(&mut self, word: usize, occurrence: usize, keyword: usize, score: f32) {
        self.matches
            .entry((word, occurrence))
            .or_default()
            .push((keyword, score));
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
        let unmatched_words = (length as usize)
            .saturating_sub(num_matches)
            .saturating_sub(unmatched_keywords);

        let penalty = unmatched_keywords as f32 * Self::KP + unmatched_words as f32 * Self::WP;

        let score = score - penalty;
        Candidate::new(field_id, score, data_id)
    }

    fn assignment(&self, num_keywords: usize) -> f32 {
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

#[derive(Debug)]
struct Inner {
    data: Data,
    fst_map: Map<Mmap>,
    inv_lists: Mmap,
    inv_list_offsets: Vec<usize>,
    lengths: Vec<u32>,
    field_to_data: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct FuzzyIndex {
    inner: Arc<Inner>,
}

fn max_edit_distance(keyword_len: usize) -> u32 {
    (keyword_len.saturating_sub(1) / 2) as u32
}

fn score_for_distance(dist: usize) -> f32 {
    match dist {
        0 => Item::ES,
        1 => Item::F1,
        2 => Item::F2,
        _ => Item::F3,
    }
}

impl FuzzyIndex {
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

    fn try_build_levenshtein(&self, keyword: &str) -> Option<Levenshtein> {
        // parse statelimit from env var, default to 10_000
        let state_limit = std::env::var("SEARCH_RDF_FUZZY_STATE_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);

        let mut max_dist = max_edit_distance(keyword.len());
        loop {
            if let Ok(lev) = Levenshtein::new_with_limit(keyword, max_dist, state_limit) {
                return Some(lev);
            } else if max_dist == 0 {
                return None;
            }
            max_dist -= 1;
        }
    }

    fn get_matches(&self, keyword: &str) -> Vec<InvList<'_>> {
        let mut matches = vec![];

        let Some(lev) = self.try_build_levenshtein(keyword) else {
            return matches; // if we can't even build a levenshtein automaton with dist 0, just return no matches
        };

        let mut stream = self.inner.fst_map.search(&lev).into_stream();

        while let Some((key_bytes, word_index)) = stream.next() {
            let matched_word = match std::str::from_utf8(key_bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let dist = strsim::levenshtein(keyword, matched_word);
            let score = score_for_distance(dist);

            let word_index = word_index as usize;
            let (inv_list, length) = self.get_inverted_list(word_index);
            matches.push(InvList {
                word: word_index,
                score,
                length,
                inv_list,
            });
        }

        matches
    }

    fn search_internal<F>(
        &self,
        query: &str,
        params: &FuzzySearchParams,
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

                    let mut entry = items.entry(field_id);
                    let item = match entry {
                        Entry::Occupied(ref mut entry) => entry.get_mut(),
                        Entry::Vacant(entry) => {
                            if let Some(worst) = worst_candidate(&top_k) {
                                let mut upper_bound_score = max_future_score;
                                upper_bound_score += inv_list.score;
                                upper_bound_score -= keyword as f32 * Item::KP;

                                let upper_bound =
                                    Candidate::new(field_id, upper_bound_score, data_id);
                                if upper_bound <= worst {
                                    skip.insert(field_id);
                                    continue;
                                }
                            }
                            entry.insert(Item::new())
                        }
                    };
                    item.update(inv_list.word, occurrence, keyword, inv_list.score);
                    let current = item.candidate(field_id, data_id, num_keywords, length);
                    let old = item.candidate.replace(current);
                    if let Some(old) = old {
                        top_k.remove(&old);
                    }

                    let Some(worst) = worst_candidate(&top_k) else {
                        top_k.insert(current);
                        continue;
                    };

                    let upper_bound = current.add(max_future_score);
                    if upper_bound <= worst {
                        skip.insert(field_id);
                        continue;
                    }

                    if current > worst {
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
pub struct FuzzySearchParams {
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

impl Default for FuzzySearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            min_score: None,
            exact: false,
        }
    }
}

impl SearchParamsExt for FuzzySearchParams {
    fn k(&self) -> usize {
        self.k
    }

    fn exact(&self) -> bool {
        self.exact
    }
}

impl Search for FuzzyIndex {
    type Data = Data;
    type Query<'q> = &'q str;
    type BuildParams = ();
    type SearchParams = FuzzySearchParams;

    fn build(data: &Self::Data, index_dir: &Path, _params: &Self::BuildParams) -> Result<()> {
        create_dir_all(index_dir)?;

        let mut inv_lists: HashMap<String, Vec<u32>> = HashMap::new();

        let mut field_to_data_file =
            BufWriter::new(File::create(index_dir.join("index.field-to-data"))?);
        let mut lengths_file = BufWriter::new(File::create(index_dir.join("index.lengths"))?);

        let total_fields = data.total_fields();
        let pb = progress_bar("Building fuzzy index", Some(total_fields as u64))?;

        let mut field_id = 0;
        for (id, fields) in data.items() {
            for field in fields.into_iter() {
                if field_id == u32::MAX {
                    return Err(anyhow!("too many fields, max {} supported", u32::MAX));
                }

                if !field.is_text() {
                    return Err(anyhow!("Fuzzy index can only be built on text fields"));
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

        pb.finish_with_message("Fuzzy index built");

        // Build FST map: word -> sequential vocabulary index
        // Also write inverted lists in the same sorted order
        let fst_path = index_dir.join("index.fst");
        let fst_file = File::create(&fst_path)?;
        let mut fst_builder = fst::MapBuilder::new(BufWriter::new(fst_file))?;

        let mut inv_list_file = BufWriter::new(File::create(index_dir.join("index.inv-lists"))?);
        let mut inv_list_offset_file =
            BufWriter::new(File::create(index_dir.join("index.inv-list-offsets"))?);
        let mut inv_list_offset = 0;

        for (word_index, (keyword, inv_list)) in
            (0_u64..).zip(inv_lists.into_iter().sorted_by(|(a, _), (b, _)| a.cmp(b)))
        {
            // Insert into FST: word -> sequential index
            fst_builder.insert(keyword.as_bytes(), word_index)?;

            // Write inverted list offset
            let inv_list_offset_bytes = u64::try_from(inv_list_offset)?.to_le_bytes();
            inv_list_offset_file.write_all(&inv_list_offset_bytes)?;

            // Write inverted list
            for field_id in &inv_list {
                inv_list_file.write_all(&field_id.to_le_bytes())?;
            }
            inv_list_offset += inv_list.len() * U32_SIZE;
        }

        fst_builder.finish()?;

        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self> {
        let fst_file = File::open(index_dir.join("index.fst"))?;
        let fst_mmap = unsafe { Mmap::map(&fst_file)? };
        let fst_map = Map::new(fst_mmap)?;

        let inv_lists = unsafe { Mmap::map(&File::open(index_dir.join("index.inv-lists"))?)? };
        let inv_list_offsets = load_usize_vec_from_u64(&index_dir.join("index.inv-list-offsets"))?;

        let lengths = load_u32_vec(&index_dir.join("index.lengths"))?;
        let field_to_data = load_u32_vec(&index_dir.join("index.field-to-data"))?;

        let inner = Arc::new(Inner {
            data,
            fst_map,
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
        "fuzzy"
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

    fn build_fuzzy_index(data: Data, index_dir: &Path) -> FuzzyIndex {
        create_dir_all(index_dir).expect("Failed to create index directory");
        FuzzyIndex::build(&data, index_dir, &()).expect("Failed to build index");
        FuzzyIndex::load(data, index_dir).expect("Failed to load index")
    }

    #[test]
    fn test_fuzzy_exact_match() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        let items = vec![Ok(DataItem::from_string_fields(
            "Q30".to_string(),
            vec![StringField {
                field_type: FieldType::Text,
                value: "United States".to_string(),
                tags: vec![],
            }],
        )
        .expect("Failed to create Item"))];

        Data::build(items, &data_dir).expect("Failed to build data");
        let data = Data::load(&data_dir).expect("Failed to load data");

        let index = build_fuzzy_index(data, &index_dir);

        let matches = index
            .search(
                "United States",
                &FuzzySearchParams {
                    k: 1,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");

        assert_eq!(matches.len(), 1);
        assert!(matches!(matches[0], Match::WithField(0, 0, score) if (score - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_fuzzy_near_match() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        let items = vec![
            Ok(DataItem::from_string_fields(
                "Q1".to_string(),
                vec![StringField {
                    field_type: FieldType::Text,
                    value: "federal state".to_string(),
                    tags: vec![],
                }],
            )
            .expect("Failed to create Item")),
            Ok(DataItem::from_string_fields(
                "Q2".to_string(),
                vec![StringField {
                    field_type: FieldType::Text,
                    value: "federated state".to_string(),
                    tags: vec![],
                }],
            )
            .expect("Failed to create Item")),
        ];

        Data::build(items, &data_dir).expect("Failed to build data");
        let data = Data::load(&data_dir).expect("Failed to load data");

        let index = build_fuzzy_index(data, &index_dir);

        // "federal" (7 chars) -> max_dist=1, "federated" is dist 3, so it won't match
        // "federal" should exactly match Q1
        let matches = index
            .search(
                "federal state",
                &FuzzySearchParams {
                    k: 10,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");

        assert!(!matches.is_empty());
        // Q1 should be the top match (exact match for both words)
        assert!(matches!(matches[0], Match::WithField(0, 0, _)));
    }

    #[test]
    fn test_fuzzy_edit_distance_scaling() {
        // (len - 1) / 2
        assert_eq!(max_edit_distance(1), 0); // 0/2
        assert_eq!(max_edit_distance(2), 0); // 1/2
        assert_eq!(max_edit_distance(3), 1); // 2/2
        assert_eq!(max_edit_distance(4), 1); // 3/2
        assert_eq!(max_edit_distance(5), 2); // 4/2
        assert_eq!(max_edit_distance(6), 2); // 5/2
        assert_eq!(max_edit_distance(7), 3); // 6/2
        assert_eq!(max_edit_distance(8), 3); // 7/2
        assert_eq!(max_edit_distance(9), 4); // 8/2
    }

    #[test]
    fn test_fuzzy_score_for_distance() {
        assert!((score_for_distance(0) - 1.0).abs() < 1e-6);
        assert!((score_for_distance(1) - 0.75).abs() < 1e-6);
        assert!((score_for_distance(2) - 0.5).abs() < 1e-6);
        assert!((score_for_distance(3) - 0.25).abs() < 1e-6);
        assert!((score_for_distance(4) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_fuzzy_short_word() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        let items = vec![
            Ok(DataItem::from_string_fields(
                "Q1".to_string(),
                vec![StringField {
                    field_type: FieldType::Text,
                    value: "cat".to_string(),
                    tags: vec![],
                }],
            )
            .expect("Failed to create Item")),
            Ok(DataItem::from_string_fields(
                "Q2".to_string(),
                vec![StringField {
                    field_type: FieldType::Text,
                    value: "bat".to_string(),
                    tags: vec![],
                }],
            )
            .expect("Failed to create Item")),
        ];

        Data::build(items, &data_dir).expect("Failed to build data");
        let data = Data::load(&data_dir).expect("Failed to load data");

        let index = build_fuzzy_index(data, &index_dir);

        // "cat" (3 chars) -> max_dist=1, "bat" is dist 1 -> matches as fuzzy
        let matches = index
            .search(
                "cat",
                &FuzzySearchParams {
                    k: 10,
                    min_score: None,
                    exact: true,
                },
            )
            .expect("Failed to find matches");

        assert_eq!(matches.len(), 2);
        // Q1 "cat" is exact match (score 1.0), Q2 "bat" is fuzzy (score 0.75)
        assert!(matches!(matches[0], Match::WithField(0, 0, score) if (score - 1.0).abs() < 1e-6));
        assert!(matches!(matches[1], Match::WithField(1, 0, score) if (score - 0.75).abs() < 1e-6));
    }
}
