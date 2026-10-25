use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub const BARK_MODEL_ID: &str = "bark-3.0-clean";

const SERVER_PY: &str = include_str!("../../resources/bark_server.py");

/// Bark (Underdog) runs in a persistent Python worker (torch + official bark_loader.py).
pub struct BarkEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    tmp_wav: std::path::PathBuf,
}

impl BarkEngine {
    pub fn load(model_file: &Path) -> Result<Self> {
        let model_dir = model_file.parent().unwrap_or(model_file);
        let python = std::env::var("HANDY_BARK_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let device = std::env::var("HANDY_BARK_DEVICE").unwrap_or_else(|_| "cpu".to_string());
        let script = std::env::temp_dir().join("handy_bark_server.py");
        std::fs::write(&script, SERVER_PY)?;
        let mut child = Command::new(&python)
            .arg("-I")
            .arg(&script)
            .arg(model_dir)
            .arg(&device)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| anyhow!("Cannot start '{}': {}", python, e))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?);
        let mut line = String::new();
        stdout.read_line(&mut line)?;
        let v: serde_json::Value = serde_json::from_str(line.trim())
            .map_err(|_| anyhow!("Bark worker failed to start (see log for Python error)"))?;
        if v.get("ready").and_then(|r| r.as_bool()) != Some(true) {
            return Err(anyhow!("Bark worker not ready"));
        }
        Ok(Self {
            child,
            stdin,
            stdout,
            tmp_wav: std::env::temp_dir().join(format!("handy_bark_{}.wav", std::process::id())),
        })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&self.tmp_wav, spec)?;
        for s in samples {
            w.write_sample(*s)?;
        }
        w.finalize()?;
        let req = serde_json::json!({ "wav": self.tmp_wav });
        writeln!(self.stdin, "{}", req)?;
        self.stdin.flush()?;
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            return Err(anyhow!("Bark worker exited"));
        }
        let v: serde_json::Value = serde_json::from_str(line.trim())?;
        if let Some(e) = v.get("error").and_then(|e| e.as_str()) {
            return Err(anyhow!("Bark: {}", e));
        }
        Ok(v.get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string())
    }

    pub fn unload(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.tmp_wav);
    }
}

impl Drop for BarkEngine {
    fn drop(&mut self) {
        self.unload();
    }
}
