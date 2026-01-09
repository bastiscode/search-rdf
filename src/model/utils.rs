use anyhow::Result;
use pyo3::{ffi::c_str, prelude::*};

// needed to be able to stop Rust code loading Python stuff with Ctrl-C
// because Python sets SIGINT handlers that capture the signal
// and prevent Rust from handling it
pub(crate) fn ignore_system_signals(py: Python) -> Result<()> {
    Ok(py.run(
        c_str!(
            r#"
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
        "#
        ),
        None,
        None,
    )?)
}
