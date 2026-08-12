"""Every tunable, one place. Phase 2 retunes these; nothing else."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Link ──────────────────────────────────────────────────────────────────
# How the device is reached. "serial" is the USB cable it is already plugged
# into for power; "ws" is WiFi, which is where this ends up once the piece is
# untethered. The firmware has a matching switch in firmware/gradi_remark/
# config.h — change both or nothing connects.
LINK = "serial"
SERIAL_PORT = None                 # None = first /dev/cu.usbmodem*

# Host -> device pacing, as a multiple of realtime audio. USB will happily
# deliver an utterance faster than the device can parse it, and the ESP32's
# CDC driver drops bytes on a full buffer rather than pushing back — which
# sounds like crackle and loses the UTT_END that ends playback. The device has
# a 10.9 s ring and a 300 ms prebuffer, so a modest lead is all it ever needs.
SERIAL_PACE = 3.0
# Bytes allowed through at full USB speed before pacing bites. Must stay well
# under the device's SERIAL_RX_BUFFER (16 KB) or the burst is itself the
# overflow. The prebuffer costs ~100 ms to fill at SERIAL_PACE, so there is no
# reason to want a big one.
SERIAL_BURST_BYTES = 4096

# Used when LINK == "ws"
BIND_HOST = "0.0.0.0"
BIND_PORT = 8765

# ── Audio ─────────────────────────────────────────────────────────────────
# Kokoro's native rate. Matching it end to end is what keeps a resampler out
# of the chain — do not change one of these without the other.
SAMPLE_RATE = 24000
PCM_CHUNK_MS = 40                  # slice size when streaming PCM to the device

# ── Motion ────────────────────────────────────────────────────────────────
IMU_HZ = 100
FRAMES_PER_MESSAGE = 5             # device batches this many MOTION frames

# ── Segmentation (PLAN 5.1) ───────────────────────────────────────────────
# Two independent gates, OR'd. The device is a ball held in a hand, not a board
# strapped to a forearm, so rotation alone is not enough to notice a movement:
# a ball lifted straight up, carried across the body, or thrown gently can
# rotate almost not at all. Gating on |w| alone made those invisible.
#
# The two gates stay separate rather than being fused into one scalar. There is
# no honest conversion between deg/s and m/s^2 — any single number would need
# an arbitrary radius — and separate gates stay independently tunable.

# Rotation gate. Validated against Phase 0 captures: at rest |w| median
# 0.11 deg/s, p99 1.73, so ARM_OFF sits ~14x above the noise floor. Arcs peaked
# 93-100, wiggles 112-133, jabs 300-481, shakes 142-496.
ARM_ON_DPS = 60.0                  # IDLE -> ONSET
ARM_OFF_DPS = 25.0                 # GESTURING -> SETTLING

# Translation gate, on |linear acceleration|. Measured on the bench at rest over
# 1405 frames: median 0.028 m/s^2, p99 0.072, max 0.098. ACCEL_OFF therefore
# sits ~3.5x above the observed floor and ACCEL_ON ~8x, which mirrors the ratio
# the rotation gate has always used.
#
# These are deliberately low. A 30 cm lift over a second peaks near 1.7 m/s^2,
# but an unhurried carry peaks nearer 0.5 — and a gate set above that recreates
# exactly the blind spot this work exists to remove. The remaining unknown is
# hand tremor, which the bench cannot show: if gestures stop settling, ACCEL_OFF
# is under the tremor of whoever is holding it and wants raising, not lowering.
ACCEL_ON_MS2 = 0.8                 # IDLE -> ONSET
ACCEL_OFF_MS2 = 0.35               # GESTURING -> SETTLING

ONSET_HOLD_MS = 80                 # sustained above either ON gate to arm
MIN_DURATION_MS = 250              # shorter than this is a twitch, discarded
SETTLE_HOLD_MS = 250               # quiet this long and the gesture is over
# A held ball invites slow, sustained movement in a way a strapped-on board did
# not — a deliberate carry across the room is a real gesture, not a runaway.
# The old 4 s cut encoded a prior that gestures are ~1 s events.
MAX_DURATION_MS = 10000            # force-cut
REFRACTORY_MS = 800                # after responding, before arming again
PREBUFFER_MS = 500                 # rolling history kept so onset isn't clipped

# ── Lifecycle (host/lifecycle.py) ─────────────────────────────────────────
# Is anyone there? The ball rests on a table until someone picks it up, and the
# machine must be completely silent until then — a piece that narrates to an
# empty room all night is unshowable.
#
# A hand cannot hold an object without micro-rotation, and that is the whole
# signal. Measured over ~1400 frames each:
#
#     on a table          gyro median 0.05 deg/s   p95 0.25
#     held still by hand  gyro median 0.78 deg/s   p95 1.58
#
# About fifteen times apart with almost no overlap. Acceleration separates them
# far less cleanly (0.028 vs 0.068) and is not used for this.
# A knock on the table produced a "visit" that lasted one second, so the bar for
# believing a person arrived is deliberately high: nearly a second of sustained
# handling, at a rate much closer to a held ball than to a quiet table. The
# greeting is delayed by that much, which nobody notices, and knocks stop
# inventing visitors.
HELD_GYRO_DPS = 0.50               # sustained rotation above this means a hand
TABLE_GYRO_DPS = 0.20              # ...and below this it is back on the table
LIFECYCLE_WINDOW_MS = 1000         # rolling window the classifier judges over
PICKUP_HOLD_MS = 900               # sustained handling before a pickup is real
# Someone holding a ball very still drops under the table threshold, and at
# 1500 ms that read as a putdown — the visit ended, a new one began, and the
# same person was greeted as the next visitor. Eight pickups produced five real
# visits. A real putdown stays down, so waiting is free.
PUTDOWN_HOLD_MS = 4000             # ...and stillness before a putdown is real
# Anything briefer than this was never a person. Belt and braces behind the
# thresholds above: the visit is not recorded and the last word is not said.
MIN_VISIT_MS = 3000

# While held. A pause is the polite moment to speak into; holding is a person
# who has picked it up and not yet decided what to do, which is a different and
# more interesting thing than an empty table.
PAUSE_MS = 700                     # quiet this long between movements is a pause
HOLDING_MS = 3000                  # quiet this long is someone deciding
HOLDING_REPEAT_MS = 15000          # ...noted again this often if it continues

# ── Free fall and impact ──────────────────────────────────────────────────
# In free fall the accelerometer measures ~0 because it is falling with the
# ball. This is the one unambiguous event an IMU can report, and it needs the
# *raw* accelerometer: the fusion's gravity estimate is meaningless while
# weightless, so anything derived from the quaternion fails exactly here.
# Validated on a real capture: two tosses read 290 ms and 360 ms airborne, and
# the flight times agreed with the integrated release speed to within 17%.
FREEFALL_MS2 = 2.5                 # |a| below this is weightless
FREEFALL_MIN_MS = 80               # ...sustained this long to count as airborne
# A catch measured 80.6 m/s^2 on one vigorous throw, and 50 was set from it.
# Then eight hand-to-hand tosses came back at 19, 27, 29, 32, 32, 48, 69 and 70
# — every one of them caught cleanly — so six were announced as fumbles. A soft
# catch is soft. The floor for "something stopped it" is now below the gentlest
# real catch observed.
IMPACT_MS2 = 15.0                  # a spike this hard is something stopping it
G = 9.80665

# ── Trajectory (host/trajectory.py) ───────────────────────────────────────
# Raw double integration diverges as t^2, but the zero-velocity update removes
# the constant-bias term that dominates that growth, so what is left degrades
# far more gently. Full confidence up to TRAJECTORY_FULL_S, decaying to nothing
# at TRAJECTORY_MAX_S.
#
# The first cut had full confidence expiring at 3 s and a slow deliberate lift —
# exactly the movement a held ball invites, and the one the old rotation gate
# could never see — came out as "barely moved, distance unclear". Reconstructing
# it perfectly and then refusing to say so is the worst of both.
TRAJECTORY_FULL_S = 2.0
TRAJECTORY_MAX_S = 6.0

# The hard limit, and it is physics rather than tuning. Acceleration scales as
# distance / time^2, so a slow movement produces almost no signal: a 30 cm lift
# over 4 seconds peaks near 0.11 m/s^2, while the tremor of a hand merely
# holding the device measures 0.12-0.27. Below that line the displacement is
# smaller than the noise it is buried in, and double integration returns
# confident nonsense — measured on a real capture at 14 cm for a 30 cm lift,
# and a downward movement reported as upward.
#
# So the reconstruction must also be judged on how far the movement rose above
# the tremor floor. A brisk gesture clears it by 20x and is measurable; a slow
# deliberate one never does, and the honest output there is "distance unclear".
# Raising the gates does not fix this and neither does better integration.
# The lower bound is evidenced: a real 30 cm lift done slowly peaked at 0.72
# (SNR 2.9) and measured 14 cm, so 3.5 rejects it with a little margin. The
# upper bound is not — it is set so a 30 cm lift taking a second (SNR 6.9,
# measured exactly right in simulation) is believed. Confirm it against a real
# brisk capture before trusting the top of this range.
TREMOR_MS2 = 0.25                  # measured: |linear accel| holding it still
TRAJECTORY_MIN_SNR = 3.5           # peak/tremor below this: no distance claimed
TRAJECTORY_GOOD_SNR = 9.0          # ...and full confidence only above this
# Residual velocity at the end of a gesture is accumulated bias, since the
# segmenter only ends a gesture once it has gone quiet. A large residual means
# that assumption broke — the gesture was force-cut mid-flight, or the ball
# never actually came to rest.
TRAJECTORY_MAX_DRIFT_MS = 1.2      # residual velocity above this = no confidence
TRAJECTORY_MIN_CONFIDENCE = 0.4    # below this, distances are not asserted

# ── Features (PLAN 5.2) ───────────────────────────────────────────────────
# Oscillation is measured on acceleration, so its noise floor is in m/s^2 and
# not the gyro's deg/s. Comfortably above the 0.25 tremor floor, and far below
# the ~3.8 peak of someone fidgeting with it.
REVERSAL_MIN_MS2 = 0.8             # ignore sign flips in the noise

# Travel directness is net displacement / path length, in space. A movement
# that ends somewhere new scores high; one that comes back scores near zero.
# This replaces the old rotational directness, which for a ball measured
# something nobody can perceive — a ball has no front, so where it ended up
# *pointing* is not a fact about what the person did.
DIRECT_RATIO = 0.70                # ended somewhere else
RETURN_RATIO = 0.30                # came back to where it started

EARLY_PEAK_FRAC = 0.35             # time-to-peak below this is "early"
LATE_PEAK_FRAC = 0.65              # above this is "late"
REPEAT_MIN_PERIODS = 2.0           # need this many cycles to call it repetition
# A hand cannot oscillate faster than about 10 Hz. Without this the
# autocorrelation locks onto a harmonic and reports nonsense — a live shake came
# back as "16.7 times a second", which is not a thing a person can do.
REPEAT_MAX_HZ = 10.0

# Size bands, in the units a person would use. Centimetres, because "raised it
# about a foot" is a description and "113 degrees of path" is not.
#
# Measured, and smaller than expected. Deliberate movements from a real capture
# spanned 13-29 cm: what a person calls "lifting it about 30 cm" is nearer 13,
# and the requested 60 came out at 29. The estimate is what is wrong there, not
# the sensor — but the *thresholds* have to match what people actually do, and
# the first guess of 15/60 put a genuine deliberate lift in the "nothing
# happened" bucket and made LARGE unreachable.
SMALL_CM = 8.0                     # below this, it did not really go anywhere
LARGE_CM = 40.0                    # above this is a big movement for this object

# ── Strokes ───────────────────────────────────────────────────────────────
# People do not hand the device discrete movements. They move it continuously,
# and a path that goes up and comes back has a net displacement of nearly zero
# however far it actually travelled — so summarising a gesture by its endpoints
# describes something nobody did. A real capture went up 79 cm and back down 31
# and was announced as "79 cm down": the distance came from the rise and the
# direction from the net.
#
# So the path is split into strokes at direction changes and described as a
# sequence. A single-stroke gesture behaves exactly as before.
STROKE_TURN_DEG = 70.0             # a heading change this sharp starts a stroke
STROKE_MIN_CM = 5.0                # shorter than this is merged into its neighbour

# Effort, from peak linear acceleration. This replaced an onset-jerk test that
# measured the wrong thing: a lift begins with a smooth ramp, so its first
# 100 ms look hesitant however committed the movement is, and the real captures
# scored brisk lifts (5) below a slow carry (32). Peak acceleration separates
# them cleanly and does not depend on exactly where the window starts —
# measured: carry 2.0, spin 3.3, lifts 5.3-9.9, shake 33.8, throw 71.3.
GENTLE_MS2 = 2.5                   # below this: barely any force in it
FORCEFUL_MS2 = 15.0                # above this: hard

# A deliberate two-handed spin measures 107 dps median. The first guess of 180
# was above anything a person actually does and so never fired at all.
SPIN_DPS = 70.0                    # sustained rate that reads as spinning
# Real spins are not clean: the same capture had an axis stability of 0.43,
# nowhere near the 1.0 an idealised spin would give. Only genuine tumbling
# should read as rolling.
TUMBLE_STABILITY = 0.30
STILL_SPEED_MS = 0.05              # slower than this is not going anywhere

# ── Salience (host/salience.py) ───────────────────────────────────────────
# How much a thing is worth saying, 0..1. Starting weights; tune against real
# captures with `replay --policy`.
SAL_BASE = 0.10                    # something happened, and that is all
# How big it was, in its own right. Salience knew about records, novelty, drama
# and repetition but nothing about magnitude, so a 276 cm six-legged sweep and a
# 5 cm twitch scored the same whenever neither was a record — and a genuinely
# large movement went unremarked, missing the bar by 0.02. A big movement is
# notable because it is big, not only because it beat something.
SAL_MAGNITUDE = 0.30
SAL_BIG_TRAVEL_CM = 200.0          # total distance covered that reads as a lot
SAL_RECORD = 0.40                  # largest, hardest, highest so far
SAL_NEW_FAMILY = 0.30              # nobody has done this kind of thing
SAL_DRAMA = 0.30                   # a throw, a catch, a hard impact
SAL_ABSENCE = 0.25                 # went where nobody has gone
SAL_FIRST_THIS_VISIT = 0.30        # new for this person, if not for the night
# The gradient. Doing the same thing again is progressively less interesting,
# and people feel that without being told — which is the whole mechanism.
SAL_REPEAT_PENALTY = 0.12
# Ceiling for a gesture too slight to name. See salience.for_gesture.
SAL_NOTHING_MUCH_MAX = 0.25
# A run long enough stops being repetition and becomes a pattern. Eight
# hand-to-hand tosses in a row produced eight decaying scores and total silence,
# when the one genuinely perceptive thing available was that it had been
# happening at all. Ignore the second, third and fourth; notice the fifth.
SAL_RUN_NOTICE = 5                 # a run this long is worth remarking on
SAL_RUN_BONUS = 0.45

# ── Policy (host/policy.py) ───────────────────────────────────────────────
# One moving bar produces variable rate, differential attention, and a gradient
# to climb. It rises whenever the machine speaks and decays back while it is
# quiet, so repetition (which scores low) stops clearing it while novelty
# (which scores high) still does.
# BAR_STEP is deliberately modest. The repetition penalty in salience.py already
# silences someone in a rut; the bar's job is pacing, not discrimination, and a
# steep step made the machine mute after any two remarks however varied the
# person was being.
BAR_BASE = 0.30                    # resting height
BAR_STEP = 0.15                    # added each time it speaks
BAR_MAX = 0.95
BAR_DECAY_S = 12.0                 # time constant back toward BAR_BASE
BAR_INTERRUPT = 0.85               # clears this and it will cut in mid-movement
PENDING_MS = 1500                  # how long a ready line waits for a lull

# ── Relational (per-session, host/server.py) ──────────────────────────────
# The corpus knows what *people* have done; this knows what has happened in the
# last minute or two. That second timescale is what makes the machine seem to
# be following someone, and it needs no identity to work — a long stillness
# resets it, which is also what a handoff looks like.
SESSION_WINDOW = 6                 # recent gestures kept for comparison
SESSION_RESET_MS = 20000           # stillness this long ends the stretch
SIMILAR_RATIO = 0.25               # within this fraction counts as "the same"

# ── Voice (PLAN 5.3) ──────────────────────────────────────────────────────
# PLAN §5.3 names llama3.2:3b. Swapped after an A/B on 30 logged gestures:
# the 3b invents objects that were never in the descriptor ("reaching for a
# light switch") and describes rather than judges. The 8b holds the register
# and compares against previous gestures. Costs ~300 ms, still inside budget.
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://127.0.0.1:11434"
# Overridable so a test persona can be swapped in without editing the real one:
#     GRADI_PERSONA=persona/plain.txt uv run python -m host.server
# persona/plain.txt is a flat readout of what the machine perceived, with none
# of the register — the fastest way to hear whether perception is right before
# judging whether the writing is.
PERSONA_PATH = Path(os.environ.get("GRADI_PERSONA",
                                   ROOT / "persona" / "persona.txt"))
# These four are env-overridable because a readout persona wants the opposite of
# what the real one wants. Reading back what just happened should be repetitive
# and deterministic — the same movement twice *should* produce the same sentence
# twice, and history plus a de-duplication retry actively fight that. Judging
# perception through a voice trying not to repeat itself measures the wrong
# thing. See persona/plain.txt.
CONTEXT_PAIRS = int(os.environ.get("GRADI_CONTEXT_PAIRS", 5))
MAX_WORDS = int(os.environ.get("GRADI_MAX_WORDS", 30))
OVERLAP_THRESHOLD = float(os.environ.get("GRADI_OVERLAP", 0.6))
TEMPERATURE = float(os.environ.get("GRADI_TEMPERATURE", 0.9))
# MAX_WORDS is a hard truncation only. Nobody waits for the device to finish
# before gesturing again, so speech length costs little and the room to say the
# whole thought is worth more than brevity — 22 was clipping the "I could not
# tell how far" that a failure most needs to end with.

# ── TTS (PLAN 5.4) ────────────────────────────────────────────────────────
TTS_MODEL = "prince-canuma/Kokoro-82M"
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0
TTS_LANG = "a"

# Kokoro comes out around -11 dBFS, which wastes most of the amp's range on a
# small speaker. Each utterance arrives as a single segment before the first
# chunk ships, so normalising costs no latency and beats a fixed gain — a fixed
# gain would have to be timid enough never to clip the loud ones.
# Drop TTS_PEAK if it sounds strained; that is distortion, not volume.
TTS_PEAK = 0.95                    # normalise each utterance to this peak
TTS_MAX_GAIN = 8.0                 # never haul a near-silent segment up this far

# ── Corpus ────────────────────────────────────────────────────────────────
# Everything the machine has watched, across every person who has picked it up.
# Delete this file to start an evening fresh; it is what makes "nobody has done
# that" a true statement rather than a bluff.
CORPUS_PATH = ROOT / "logs" / "corpus.jsonl"

# ── Logging (PLAN 5.6) ────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
