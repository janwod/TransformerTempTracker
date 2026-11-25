"""AISE 4010 project (Group 07): run the exported oil temperature models with NumPy only, no TensorFlow.
Usage:  python3 jetson_forecast.py          check, score, time and forecast with every exported model
        python3 jetson_forecast.py 3000     time 3000 forecasts per model (watch it with: sudo tegrastats)
"""
import json, math, os, platform, sys, time
import numpy as np
try:
    import resource                                    # macOS fallback for the peak RAM line
except ImportError:
    resource = None

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def erf(x):
    # NumPy has no erf. Abramowitz & Stegun 7.1.26, max error 1.5e-7, far below float32 noise here.
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))    # exact GELU, same as Keras' default

ACT = {'tanh': np.tanh, 'relu': lambda x: np.maximum(x, 0.0), 'linear': lambda x: x,
       'sigmoid': sigmoid, 'gelu': gelu}

def dense(x, W, b, act='linear'):
    return ACT[act](x @ W + b)

def recurrent(cell, X, W, U, b, act):
    """One recurrent layer, one hour after the other. Returns the hidden state of every step."""
    n = U.shape[0]                                     # number of units
    h = np.zeros((X.shape[0], n), dtype=np.float32)    # hidden state, starts at zero
    c = np.zeros_like(h)                               # LSTM cell state
    states = []
    for t in range(X.shape[1]):
        x = X[:, t]
        if cell == 'SimpleRNN':
            h = act(x @ W + h @ U + b)
        elif cell == 'LSTM':                           # Keras stores 4 blocks side by side: i, f, c~, o
            a = x @ W + h @ U + b
            i, f, o = sigmoid(a[:, :n]), sigmoid(a[:, n:2 * n]), sigmoid(a[:, 3 * n:])
            c = f * c + i * act(a[:, 2 * n:3 * n])
            h = o * act(c)
        else:                                          # GRU, 3 blocks: z, r, h~ (Keras: z = share of OLD state kept)
            ax, ah = x @ W + b[0], h @ U + b[1]
            z = sigmoid(ax[:, :n] + ah[:, :n])
            r = sigmoid(ax[:, n:2 * n] + ah[:, n:2 * n])
            h = z * h + (1.0 - z) * act(ax[:, 2 * n:] + r * ah[:, 2 * n:])
        states.append(h)
    return np.stack(states, axis=1)                    # (windows, hours, units)

def layer_norm(x, gamma, beta, eps):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

def attention(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
    """Multi-head self-attention, same maths as Keras' MultiHeadAttention.
    Wq/Wk/Wv: (d_model, heads, key_dim), Wo: (heads, key_dim, d_model)."""
    key_dim = Wq.shape[-1]
    q = np.einsum('btd,dhk->bthk', x, Wq) + bq
    k = np.einsum('btd,dhk->bthk', x, Wk) + bk
    v = np.einsum('btd,dhk->bthk', x, Wv) + bv
    scores = np.einsum('bthk,bshk->bhts', q / math.sqrt(key_dim), k)    # how much each token looks at each other one
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)                             # softmax over the tokens being looked at
    mixed = np.einsum('bhts,bshk->bthk', scores, v)
    return np.einsum('bthk,hkd->btd', mixed, Wo) + bo

def predict(m, w, X, ot_idx):
    """Run one exported model on scaled windows (windows, 96, 7). Returns the scaled 24-hour forecast."""
    layers = [(L, [w['%d_%d' % (k, j)] for j in range(L['n'])]) for k, L in enumerate(m['layers'])]
    last = X[:, -1:, :]
    last_ot = last[:, 0, ot_idx][:, None]
    x = X - last                                       # same normalisation as in the notebook
    if m['arch'] == 'mlp':
        x = x.reshape(len(x), -1)                      # Flatten: 96 x 7 = 672 inputs
        for L, (W, b) in layers:
            x = dense(x, W, b, L['act'])
    elif m['arch'] == 'rnn':
        for L, wts in layers:
            if L['kind'] == 'Dense':
                x = dense(x, wts[0], wts[1], L['act'])
            else:
                H = recurrent(L['kind'], x, *wts, act=ACT[L['act']])
                x = H if L['seq'] else H[:, -1]        # return_sequences=False keeps only the last hour
    else:                                              # transformer
        p = m['patch_len']
        x = x.reshape(len(x), x.shape[1] // p, p * x.shape[2])        # one token per patch (one day)
        it = iter(layers)
        L, (W, b) = next(it)
        x = dense(x, W, b) + w['pos']                  # token embedding + positional encoding
        blocks = (len(layers) - 3) // 5                # each block: LN, attention, LN, Dense, Dense
        for _ in range(blocks):
            L, (g, be) = next(it)
            h = layer_norm(x, g, be, L['eps'])
            _, att = next(it)
            x = x + attention(h, *att)                 # residual connection
            L, (g, be) = next(it)
            h = layer_norm(x, g, be, L['eps'])
            L1, (W1, b1) = next(it)
            L2, (W2, b2) = next(it)
            x = x + dense(dense(h, W1, b1, L1['act']), W2, b2, L2['act'])
        L, (g, be) = next(it)
        x = layer_norm(x, g, be, L['eps']).reshape(len(x), -1)        # final LayerNorm, then Flatten
        L, (W, b) = next(it)
        x = dense(x, W, b, L['act'])
    return x + last_ot                                 # predicted change + last OT

def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(here, 'project_models.npz'))
    meta = json.loads(str(data['meta']))
    ot = meta['ot_idx']
    mean, std = np.array(meta['mean'], np.float32), np.array(meta['std'], np.float32)
    to_c = lambda z: z * std[ot] + mean[ot]
    X, y = data['test_X'].astype(np.float32), data['test_y']
    latest = ((data['last_raw'] - mean) / std)[None].astype(np.float32)    # scale raw readings like in training

    board = '/proc/device-tree/model'                  # only exists on boards such as the Jetson
    device = open(board).read().strip('\x00\n ') if os.path.exists(board) else platform.machine()
    print('%s | %s | Python %s | NumPy %s | %d CPU cores' % (platform.node(), device, platform.python_version(),
                                                             np.__version__, os.cpu_count()))
    print('last hour in the data: %s (OT %.1f °C) -> forecast for the next 24 hours' % (meta['last_hour'], meta['last_ot']))
    print('%d test windows, timing %d single-window forecasts per model\n' % (len(X), runs))
    print('%-26s %10s %18s %14s %12s %10s %10s' % ('model', 'parameters', 'max |NumPy-Keras|', 'test MAE (°C)',
                                                    'ms/forecast', '+1 h (°C)', '+24 h (°C)'))
    for key, m in meta['models'].items():
        w = {k[len(key) + 1:]: data[k] for k in data.files if k.startswith(key + '/')}
        check = w['keras_c']
        diff = np.abs(to_c(predict(m, w, X[:len(check)], ot)) - check).max()     # same answers as Keras?
        pred = np.concatenate([predict(m, w, X[i:i + 256], ot) for i in range(0, len(X), 256)])
        mae = np.abs(to_c(pred) - to_c(y)).mean()                         # accuracy computed on the device
        for _ in range(10):
            predict(m, w, X[:1], ot)                                          # warm-up
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            predict(m, w, X[:1], ot)                                          # one window at a time, as in real use
            times.append(time.perf_counter() - start)
        fc = to_c(predict(m, w, latest, ot))[0]
        print('%-26s %10s %15.1e °C %14.3f %12.3f %10.1f %10.1f' % (m['name'], format(m['params'], ','), diff, mae,
                                                                  1000 * np.median(times), fc[0], fc[-1]))
    print('\npeak RAM of this program (all models + test data + Python): %s' % peak_ram())

def peak_ram():
    """Peak memory of this process only. /proc/self/status resets when the program starts, unlike
    getrusage, which on Linux also counts the process that launched it (e.g. a Jupyter kernel)."""
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmHWM:'):
                return '%.0f MB' % (int(line.split()[1]) / 1024)
    except OSError:
        pass
    if resource:                                       # macOS fallback (reports bytes)
        return '%.0f MB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
    return 'not available on this OS'

if __name__ == '__main__':
    main()
