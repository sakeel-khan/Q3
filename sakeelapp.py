"""
app.py  —  Q3B: Sonic Signatures — Interactive Music Identifier
=================================================================
Run with:  streamlit run app.py

• Reads the fingerprint database built by EE200_Q3A.ipynb (fingerprint_db.pkl)
• Single-clip mode  : upload a clip → spectrogram + constellation + offset histogram + result
• Batch mode        : upload multiple clips → results.csv download
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import pickle
import os
import io
import csv
import tempfile
from collections import defaultdict
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion

# ──────────────────────────────────────────────────────────────────────────────
# Parameters  (must match the notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────
SR          = 22050
N_FFT       = 4096
HOP_LENGTH  = 512
PEAK_NEIGH  = 20
PEAK_AMP_MIN= 10
MAX_PEAKS   = 100
FAN_VALUE   = 5
MIN_HASH_DT = 0
MAX_HASH_DT = 100
DB_PATH     = 'fingerprint_db.pkl'

# Frequency band (must match notebook)
import numpy as _np
FREQ_MIN_HZ = 300
FREQ_MAX_HZ = 2000
_all_freqs  = _np.fft.rfftfreq(N_FFT, d=1.0/SR)
FREQ_BIN_LO = int(_np.searchsorted(_all_freqs, FREQ_MIN_HZ))
FREQ_BIN_HI = int(_np.searchsorted(_all_freqs, FREQ_MAX_HZ))

# ──────────────────────────────────────────────────────────────────────────────
# Core DSP functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectrogram(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    D    = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db


def get_peaks(S_db, neigh=PEAK_NEIGH, amp_min=PEAK_AMP_MIN):
    # restrict to useful frequency band (matches notebook fix)
    S_db = S_db[FREQ_BIN_LO:FREQ_BIN_HI, :]
    neighbourhood = maximum_filter(S_db, size=neigh)
    local_max     = (S_db == neighbourhood)
    background    = (S_db == np.min(S_db))
    struct        = generate_binary_structure(2, 1)
    eroded        = binary_erosion(background, structure=struct, border_value=1)
    detected      = local_max ^ eroded
    detected      = detected & (S_db > (S_db.min() + amp_min))
    freq_idx, time_idx = np.where(detected)
    freq_idx += FREQ_BIN_LO   # shift back to global FFT bin coordinates
    return time_idx, freq_idx


def generate_hashes(t_peaks, f_peaks, fan_value=FAN_VALUE,
                    min_dt=MIN_HASH_DT, max_dt=MAX_HASH_DT):
    hashes = []
    order  = np.argsort(t_peaks)
    t_s, f_s = t_peaks[order], f_peaks[order]
    for i in range(len(t_s)):
        j, partners = i + 1, 0
        while j < len(t_s) and partners < fan_value:
            dt = int(t_s[j]) - int(t_s[i])
            if dt < min_dt:
                j += 1; continue
            if dt > max_dt:
                break
            hashes.append(((int(f_s[i]), int(f_s[j]), dt), int(t_s[i])))
            j += 1; partners += 1
    return hashes


def match_query(query_hashes, db, id2song, top_n=5):
    offset_counts = defaultdict(lambda: defaultdict(int))
    for h, q_t in query_hashes:
        if h in db:
            for song_id, db_t in db[h]:
                offset_counts[song_id][db_t - q_t] += 1

    scores = {sid: max(off.values()) for sid, off in offset_counts.items()}
    if not scores:
        return None, {}, {}

    ranked    = sorted(scores.items(), key=lambda x: -x[1])
    best_id   = ranked[0][0]
    best_name = id2song[best_id]
    top_ids   = [sid for sid, _ in ranked[:top_n]]

    named_scores = {id2song[sid]: sc for sid, sc in ranked}
    histograms   = {id2song[sid]: dict(offset_counts[sid]) for sid in top_ids}
    return best_name, named_scores, histograms


def fingerprint_audio(y):
    """Full pipeline: audio → hashes + intermediates."""
    S_db     = compute_spectrogram(y)
    t_p, f_p = get_peaks(S_db)
    hashes   = generate_hashes(t_p, f_p)
    return hashes, S_db, t_p, f_p

# ──────────────────────────────────────────────────────────────────────────────
# Database loading  (cached so it loads only once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='Loading fingerprint database…')
def load_database():
    if not os.path.exists(DB_PATH):
        return None, None
    with open(DB_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['db'], data['id2song']

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_spectrogram(S_db, title='Spectrogram'):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db, sr=SR, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='log', ax=ax, cmap='magma'
    )
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_constellation(t_p, f_p, S_db, title='Constellation'):
    times_sec = librosa.frames_to_time(t_p, sr=SR, hop_length=HOP_LENGTH)
    freqs_hz  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[f_p]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    librosa.display.specshow(S_db, sr=SR, hop_length=HOP_LENGTH,
                              x_axis='time', y_axis='log',
                              ax=axes[0], cmap='magma')
    axes[0].scatter(times_sec, freqs_hz, color='cyan', s=4, alpha=0.7)
    axes[0].set_title('Spectrogram + peaks')

    axes[1].scatter(times_sec, freqs_hz, s=3, alpha=0.6, color='steelblue')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title(title)

    plt.tight_layout()
    return fig


def plot_offset_histogram(histograms, true_name, scores, top_n=4):
    songs_to_plot = list(histograms.keys())[:top_n]
    n = len(songs_to_plot)
    if n == 0:
        return None

    cols = min(n, 2)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, song in enumerate(songs_to_plot):
        h      = histograms[song]
        score  = scores.get(song, 0)
        color  = 'green' if song == true_name else 'salmon'
        label  = '✓ TRUE MATCH' if song == true_name else '✗ wrong'
        axes[i].bar(list(h.keys()), list(h.values()), width=5,
                    color=color, alpha=0.85)
        axes[i].set_title(f'{song[:30]}\n(score={score}) [{label}]', fontsize=9)
        axes[i].set_xlabel('Offset (frames)')
        axes[i].set_ylabel('Hash count')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Offset histograms — peak = alignment at correct offset',
                 fontsize=11)
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Sonic Signatures — Music Identifier',
    page_icon='🎵',
    layout='wide'
)

st.title('🎵 Sonic Signatures — Music Fingerprint Identifier')
st.caption('EE200 Q3B | Shazam-style audio fingerprinting using STFT spectrograms & constellation hashing')

# ── DB status ─────────────────────────────────────────────────────────────────
db, id2song = load_database()

if db is None:
    st.error(
        f'**Database not found** (`{DB_PATH}`).\n\n'
        'Please run the notebook `EE200_Q3A.ipynb` first to build '
        'and save the fingerprint database, then place `fingerprint_db.pkl` '
        'in the same directory as this app.'
    )
    st.stop()

st.success(
    f'✅ Database loaded — **{len(id2song)} songs**, '
    f'**{len(db):,} unique hashes**'
)

with st.expander('📂 Songs in database'):
    for sid, name in sorted(id2song.items()):
        st.write(f'• {name}')

st.divider()

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    'Select mode',
    ['🎧 Single-clip identification', '📦 Batch identification (→ results.csv)'],
    horizontal=True
)

# ═════════════════════════════════════════════════════════════════════════════
# SINGLE-CLIP MODE
# ═════════════════════════════════════════════════════════════════════════════
if mode.startswith('🎧'):
    st.subheader('Single-clip Mode')
    st.write('Upload a short audio clip (MP3 / WAV / FLAC). '
             'The app will identify it and show all intermediate steps.')

    uploaded = st.file_uploader(
        'Upload query clip', type=['mp3', 'wav', 'flac', 'ogg', 'm4a']
    )

    clip_dur = st.slider('Clip duration to use (seconds)', 5, 30, 10)

    if uploaded is not None:
        # save to temp file so librosa can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner('Analysing audio…'):
            try:
                y, _ = librosa.load(tmp_path, sr=SR, mono=True, duration=clip_dur)
                hashes, S_db, t_p, f_p = fingerprint_audio(y)
                best, scores, histograms = match_query(hashes, db, id2song)
            except Exception as e:
                st.error(f'Error processing audio: {e}')
                os.unlink(tmp_path)
                st.stop()
            finally:
                os.unlink(tmp_path)

        # ── Result banner ──────────────────────────────────────────────────
        if best:
            top_score = scores.get(best, 0)
            st.success(f'## 🎶 Identified: **{best}**   (score = {top_score})')
        else:
            st.error('No match found — try a longer clip or different audio.')

        # ── Top-5 scores table ─────────────────────────────────────────────
        st.subheader('Top-5 candidates')
        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
        table_data = {'Song': [s for s, _ in top5],
                      'Score (max offset hits)': [sc for _, sc in top5]}
        st.table(table_data)

        # ── Step 1: Spectrogram ────────────────────────────────────────────
        st.subheader('Step 1 — Spectrogram')
        st.write('Short-Time Fourier Transform: time on x-axis, log-frequency on y-axis, '
                 'amplitude (dB) as brightness.')
        fig_spec = plot_spectrogram(S_db, title=f'Spectrogram of "{uploaded.name}"')
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        # ── Step 2: Constellation ──────────────────────────────────────────
        st.subheader('Step 2 — Constellation (local peak maxima)')
        st.write(f'{len(t_p)} peaks extracted · {len(hashes)} pair-hashes generated')
        fig_con = plot_constellation(t_p, f_p, S_db, title='Constellation of peaks')
        st.pyplot(fig_con)
        plt.close(fig_con)

        # ── Step 3: Offset histograms ──────────────────────────────────────
        st.subheader('Step 3 — Offset histogram')
        st.write('For each candidate song, the histogram of time offsets at which '
                 'its hashes align with the query. A true match creates a single '
                 'tall spike; false matches produce only scattered noise.')
        if histograms:
            fig_hist = plot_offset_histogram(histograms, best, scores)
            if fig_hist:
                st.pyplot(fig_hist)
                plt.close(fig_hist)
        else:
            st.info('No hash matches found in the database.')

# ═════════════════════════════════════════════════════════════════════════════
# BATCH MODE
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.subheader('Batch Mode')
    st.write(
        'Upload one or more query clips. '
        'The app will identify each one and let you download a `results.csv` '
        'with columns `filename, prediction`.'
    )

    uploaded_files = st.file_uploader(
        'Upload query clips', type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
        accept_multiple_files=True
    )

    clip_dur_batch = st.slider('Clip duration to use per file (seconds)', 5, 30, 10)

    if uploaded_files and st.button('🚀 Run batch identification'):
        results = []
        progress = st.progress(0, text='Processing…')
        status_box = st.empty()

        for i, uf in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files),
                              text=f'Processing {uf.name} …')
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uf.name)[1]
            ) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            try:
                y, _ = librosa.load(tmp_path, sr=SR, mono=True,
                                     duration=clip_dur_batch)
                hashes, S_db, t_p, f_p = fingerprint_audio(y)
                best, scores, _ = match_query(hashes, db, id2song)
                prediction = best if best else 'UNKNOWN'
            except Exception as e:
                prediction = f'ERROR: {e}'
            finally:
                os.unlink(tmp_path)

            fname_no_ext = os.path.splitext(uf.name)[0]
            results.append((uf.name, fname_no_ext, prediction))
            status_box.write(f'`{uf.name}` → **{prediction}**')

        progress.progress(1.0, text='Done!')

        # ── Show results table ─────────────────────────────────────────────
        st.subheader('Results')
        st.table({'filename': [r[0] for r in results],
                  'prediction': [r[2] for r in results]})

        # ── Build CSV (exact format required by coursework) ────────────────
        csv_buf = io.StringIO()
        writer  = csv.writer(csv_buf)
        writer.writerow(['filename', 'prediction'])
        for _, fname_no_ext, pred in results:
            writer.writerow([fname_no_ext, pred])
        csv_bytes = csv_buf.getvalue().encode('utf-8')

        st.download_button(
            label='⬇️  Download results.csv',
            data=csv_bytes,
            file_name='results.csv',
            mime='text/csv'
        )

        st.info(
            'The `prediction` column contains the **matched song filename without extension**, '
            'as required by the evaluation script.'
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    'EE200 Q3B · Sonic Signatures · '
    'Built with Streamlit, librosa, NumPy & SciPy'
)
