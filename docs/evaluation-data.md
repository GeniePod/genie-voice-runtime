# Evaluation Data Sources

This repo can use public datasets for voice-runtime evaluation, but the data
must be mapped to the right boundary. `genie-voice-runtime` measures audio and
transcript quality. `genie-claw` measures tool-call accuracy, memory retrieval,
home-state reasoning, safety policy, and BFCL score.

Do not add large public datasets to git. Download or generate them under a
local ignored directory such as `tests/eval/local/`, keep attribution metadata
with the local copy, and commit only small adapters, manifests, or synthetic
fixtures.

## Usability Matrix

| Dataset | License status | Use in `genie-voice-runtime` | Use in `genie-claw` / home agent |
|---------|----------------|------------------------------|-----------------------------------|
| [Home Assistant Intents](https://github.com/OHF-Voice/intents) | CC BY 4.0 in the source repo | Use as transcript text for STT normalization and command-phrase robustness. Do not evaluate device action correctness here. | Strong fit for BFCL `home_control`, `home_status`, timer, media, weather, and slot tests. |
| [CASAS Smart Home Data Sets](https://casas.wsu.edu/datasets/) | Current Zenodo CASAS records are CC BY 4.0; verify per record before importing. | Not a voice dataset. Use only to replay context labels around transcripts if a voice-session test needs realistic presence/activity context. | Best public source for real home sensor timelines, presence, action history, and activity-state questions. |
| [REFIT Electrical Load Measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/) | CC BY 4.0 | Not a voice dataset. | Use for deterministic energy/device-state questions such as "what is using power" and appliance activity checks. |
| [UK-DALE](https://jack-kelly.com/data/) | CC BY 4.0 | Not a voice dataset. | Use for longer appliance electricity traces, energy summaries, and power anomaly tests. |
| [SLURP](https://github.com/pswietojanski/slurp) | Text annotations are CC BY 4.0. Audio hosted on Zenodo is CC BY-NC 4.0 unless a separate license is obtained. | Use text for assistant utterance robustness. Do not put SLURP audio into product CI or commercial distribution without license clearance. | Partial fit for assistant intent mapping. Not home-specific, so it should not dominate BFCL/home fixtures. |
| [MASSIVE](https://github.com/alexa/massive) | Dataset is CC BY 4.0; repo code is Apache 2.0. | Use text for multilingual transcript robustness and slot extraction from noisy STT-like text. | Partial fit for multilingual tool-routing and slot tests after mapping intents to Genie typed tools. |
| [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) | CC BY 4.0 for dataset content | Use for wake-word, keyword spotting, command-word false-positive checks, and Jetson latency/memory smoke tests. | Not enough semantic structure for BFCL beyond very small command-word smoke fixtures. |

## Project Policy

- Prefer CC BY 4.0 sources that allow product use with attribution.
- Keep source license, citation, download URL, version, and conversion script
  together in the local eval manifest.
- Treat noncommercial data as research-only unless the project obtains a
  separate license. This applies to SLURP audio.
- Do not train or tune agent prompts by dumping large dataset text into the
  runtime prompt. Convert datasets into small typed fixtures and measure the
  score.
- For Jetson Orin 8GB validation, keep generated eval subsets small enough to
  run locally without memory pressure hiding runtime bugs.

## Recommended Mapping

1. Use Home Assistant Intents to generate GenieClaw BFCL cases for typed home
   tools, then run those predictions through `genie-ctl bfcl-score`.
2. Use CASAS timelines to synthesize realistic home context before a transcript:
   presence, room activity, device history, and sensor state.
3. Use REFIT and UK-DALE to build deterministic energy/state tests for home
   memory and typed device-status tools.
4. Use MASSIVE and SLURP text to stress transcript normalization, misspellings,
   multilingual phrasing, and assistant-style slot extraction.
5. Use Google Speech Commands only for voice-runtime wake/keyword and low-level
   audio robustness, not as a home-agent correctness benchmark.

The expected result is not a single public benchmark. It is a layered eval
suite: audio robustness in `genie-voice-runtime`, typed tool-call accuracy in
`genie-claw`, and real home state replay through CASAS/energy traces.
