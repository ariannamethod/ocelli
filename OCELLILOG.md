# OCELLILOG

Working log of the ocelli engine. Newest entries at the bottom. Every claim here
carries the command, file, or number it came from — this log is read by other
machines working on the same repo, so it records what changed in the system, not
who was in the room.

---

## 2026-07-27 — repo founded, engine moved in

The engine was developed inside a fork of a quiz application and has outgrown it:
it is an inference organ, not a game feature. Moved out as its own project.

**What moved:** `ocelli.c` (orchestrator, text decoder, vision splice — formerly
`smolvlm.c`), `vision.c/.h` (idefics3 preprocessing, SigLIP tower, pixel-shuffle
connector), `gguf.c/.h`, `bpe.c/.h` (GPT-2 byte-level BPE, special-token aware),
vendored `notorch.c/.h`, `notorch_vision.h`, `stb_image.h`, and the `eye` CLI.
Added: `Makefile` (BLAS on by default, `BLAS=0` for a portable build, `asan`
target), `.gitignore` (weights and binaries never committed), README.

**Own code vs vendored:** 1918 lines of engine C, 14349 lines vendored from the
notorch canon (`wc -l`).

### State carried over

- Two model generations run on this engine: SmolVLM-256M-Instruct and
  SmolVLM2-500M-Video-Instruct with a contract fine-tune (the Yent eye).
- Architecture is read from the file, never assumed: for the 500M weights the
  loader reports `E=960 H=15 KV=5 FFN=2560 V=49280 L=32 HD=64 rope=100000
  rms=1e-05`, matching `llama-mtmd-cli` metadata for the same GGUF.
- Vision tower reports `L=12 D=768 img=512 patch=16 patches=1024 eps=1e-06`,
  producing `[1024,768]` hidden states and `[64,960]` visual embeddings, NaN=0.
- Special tokens are identical across both generations (`<image>`=49190,
  `<fake_token_around_image>`=49189, `<global-img>`=49152,
  `<end_of_utterance>`=49279), prompt layout is 84 tokens.

### Fixed on the way in: MLP role must come from shape, not name

Loading the 500M weights produced a coherent text decoder but garbage
descriptions, while the reference runtime handled the same two files correctly.
Cause: the vision tower had a hardcoded `ffn_up` ↔ `ffn_down` swap, introduced
because one GGUF converter mirrors those names. A newer converter does not
mirror them, so the swap corrupted all 12 SigLIP layers.

Ground truth is the bias length in the file (`llama-gguf`): the older mmproj has
`ffn_down.bias = 12288 B`, the newer one has `ffn_up.bias = 12288 B`, and the
bias values themselves match between files to the digit (`-0.894531`,
`-1.398438`, `0.099609`) — the same SigLIP weights under mirrored names.

Fix: `siglip_load` now resolves fc1/fc2 per layer from the bias element count
(fc1 ⇔ bias == ffn); a length matching neither `ffn` nor `hidden` is a loud
load failure. Both conventions work; the older generation still reproduces its
stored reference outputs word for word.

### Cost, measured

Phone-class A18 Pro, 500M weights, one 512×512 frame, 84-token prompt:

| weights | prompt | generation | peak RSS |
|---|---|---|---|
| f16 | 3.6–4.1 tok/s | 3.8–3.9 tok/s | 1.0–1.4 GB |
| Q8_0 | 30.3–33.3 tok/s | 26.0–34.6 tok/s | 1.8–2.5 GB |

The 8× gap is not arithmetic. Q8_0 is dequantised once at load; the f16 path
expands every weight to f32 before every matmul, so ~500M weights are dequantised
per token. Descriptions agree between the two paths on most frames and diverge on
near-tie tokens, deterministically.

### Objective probe: small-text reading, 14 frames

Comparing free-form descriptions against a reference runtime is not a truth
test — both can invent clothing that is not in the frame, and on one portrait
both did (ours "white shirt", the reference "purple shirt", while the crop shows
no shirt at all). So the probe was narrowed to something with a knowable answer.

All 14 portraits carry the same watermark, `StyleGAN2 (Karras et al.)`, verified
by eye on two of them. Same weights, same prompt, same frames:

| runtime | watermark read correctly | corrupted readings |
|---|---|---|
| this engine | 0 / 14 | 2 (`Sylvanian`, `Icac`) |
| reference runtime | 5 / 14 | 3 (`StyGAN2`, `StyleGAN Karras et al.`, `MEG222`) |

Neither is good at it; the gap is one-sided. Since the weights are identical,
the loss is in our vision path — f32 tower against the reference's f16/mixed,
and possibly resize or normalisation upstream of it. This is the acceptance gate
for any accuracy work on the tower: the count on these 14 frames must rise.

Corollary, recorded so it is not relitigated: fine-tuning the weights is the
wrong instrument for this defect. It would compensate a runtime inaccuracy by
altering the model, welding our own error into the weights. Data first, and a
labelled set of real frames, before any training is discussed.

### Root cause of the blindness: one frame where the reference sends seventeen

The small-text failure was not the resize filter and not the weights. The
engine encoded a single global frame — 64 visual tokens — while the reference
runtime tiles the image and encodes many. Slice counts taken from its own log:

| frame | size | reference slices |
|---|---|---|
| portrait | 1024×1024 | 17 |
| terminal capture | 640×448 | 13 |
| app screenshot | 896×727 | 17 |

The rule reproduces on all three: scale the **longest edge to 2048** (up as well
as down — it was implemented here as a cap only), cut a `ceil(W/512) × ceil(H/512)`
grid of 512 tiles, append one global frame. 1024×1024 → 4×4+1 = 17.

A tiling path existed behind an opt-in flag but was dead: the preprocessor cut
the frames while the engine still encoded only the first one and spliced 64
tokens — and the first frame is the top-left tile, not the global view, so the
flag made descriptions worse, not better. The flag was calibrated in June
against a reference build that did not tile at the time; that calibration
expired with the upstream change and nobody re-checked it.

Now: every frame goes through the tower, the prompt carries per-tile
`<row_i_col_j>` markers row-major with a newline per tile row followed by the
global frame, and all `nf × 64` embeddings are spliced. Layout is confirmed by
token count rather than by reading someone's source — 1143 prompt tokens on a
4×4 grid, the exact prompt length the reference reports for the same image.

**Result on the frame that used to fail:** the watermark, previously read as
`Sylvanian`, now comes out verbatim: `StyleGAN2 (Karras et al.)`.

**Price:** `prompt 1143 tok in 374955 ms (3.0 tok/s) | gen 40 tok in 15477 ms |
peak RSS 1110 MB` — seventeen tower passes and 1143 decoder forwards. Memory is
unchanged; time is six minutes per frame on the f16 path. Packed kernels are no
longer an optimisation, they are what makes this usable.

**Not fixed by tiling, and not ours to fix in code:** both runtimes invent
clothing that is not in the frame (ours "glasses and a necklace", the reference
"purple shirt", on a crop showing neither). That is a data question.

### Open

- Hot path still runs dense f32 through BLAS. The vendored notorch already ships
  packed kernels (`nt_qmatvec`, row kernels for `q4_0/q8_0/q5_0/q4_k/q6_k`) that
  dequantise per block in registers with no f32 resident. Moving the decoder and
  the tower onto them is the path to phone-sized memory; acceptance is a parity
  run against the current Q8 path with named divergences, plus speed and RSS on
  the same frames.
- Not bit-exact against the reference runtime (f32 tower vs f16/mixed there).
- Small phone-sized text is a blind spot.
- Camera input (V4L2 / phone) is not implemented; the eye currently eats files.
