# Genie Voice Runtime

External voice runtime for the Genie ecosystem.

`genie-voice-runtime` owns the audio pipeline:

- wake word and push-to-talk activation
- voice activity detection
- capture and playback device handling
- speech-to-text
- text-to-speech
- acoustic echo control and denoise stages
- streaming voice session events

It does **not** own the agent.

`genie-claw` remains the agent layer: prompt policy, memory, tools, smart-home
intent, safety gates, audit, and conversation state. The voice runtime turns
audio into text and text back into audio; GenieClaw decides what the user meant
and what actions are allowed.

## Boundary

```text
microphone / speaker
        |
        v
genie-voice-runtime
  wake, VAD, STT, TTS, audio streaming
        |
        | transcript / spoken reply events
        v
genie-claw
  agent policy, memory, tools, smart-home actions
        |
        v
genie-home-runtime / Home Assistant
```

The protocol types in this crate are the stable contract between GenieClaw and
the voice runtime. The runtime implementation can evolve without forcing
GenieClaw to own audio details.

## Design Rules

- No agent prompts, memory policy, tool routing, or smart-home authorization in
  this repo.
- No Home Assistant or `genie-home-runtime` device logic in this repo.
- No LLM provider logic in this repo.
- Voice hardware should be portable across Jetson, Raspberry Pi, other SBCs,
  Linux laptops, and development machines where possible.
- Jetson / GeniePod Home remains the flagship tested deployment.

## Status

Initial boundary scaffold. The current production voice code still lives in
`genie-claw` and will move here in slices:

1. protocol and config contract
2. STT/TTS process wrappers
3. streaming session events
4. wake/VAD/audio-device handling
5. GenieClaw external runtime client
6. removal of the internal GenieClaw voice pipeline
