# Giải Thích Chuyên Sâu: Tại Sao Các Kỹ Thuật Này Hoạt Động?

> Tài liệu này đi sâu vào **lý do toán học** và **trực giác** đằng sau mỗi quyết định thiết kế trong notebook. Mỗi phần trả lời câu hỏi: **"Tại sao nó lại hoạt động như vậy?"**

---

## Mục Lục

1. [Orthogonal Initialization — Tại Sao Ổn Định Gradient?](#1-orthogonal-initialization--tại-sao-ổn-định-gradient)
2. [Tại Sao LSTM Giải Quyết Được Vanishing Gradient?](#2-tại-sao-lstm-giải-quyết-được-vanishing-gradient)
3. [Tại Sao GRU Đơn Giản Hơn Nhưng Vẫn Hiệu Quả?](#3-tại-sao-gru-đơn-giản-hơn-nhưng-vẫn-hiệu-quả)
4. [Tại Sao Self-Attention Dùng tanh + Linear (Additive Attention)?](#4-tại-sao-self-attention-dùng-tanh--linear-additive-attention)
5. [Tại Sao Scaled Dot-Product Chia Cho √d_k?](#5-tại-sao-scaled-dot-product-chia-cho-d_k)
6. [Tại Sao LayerNorm Thay Vì BatchNorm Cho Transformer?](#6-tại-sao-layernorm-thay-vì-batchnorm-cho-transformer)
7. [Tại Sao Residual Connections Giúp Gradient Flow Tốt?](#7-tại-sao-residual-connections-giúp-gradient-flow-tốt)
8. [Tại Sao AdamW Tách Rời Weight Decay?](#8-tại-sao-adamw-tách-rời-weight-decay)
9. [Tại Sao Warmup Giúp Huấn Luyện Ổn Định?](#9-tại-sao-warmup-giúp-huấn-luyện-ổn-định)
10. [Tại Sao Cosine Decay Tốt Hơn Step Decay?](#10-tại-sao-cosine-decay-tốt-hơn-step-decay)
11. [Tại Sao Gradient Clipping Hoạt Động?](#11-tại-sao-gradient-clipping-hoạt-động)
12. [Tại Sao BCEWithLogitsLoss Ổn Định Số Học?](#12-tại-sao-bcewithlogitsloss-ổn-định-số-học)
13. [Tại Sao Positional Encoding Dùng Sin/Cos?](#13-tại-sao-positional-encoding-dùng-sincos)
14. [Tại Sao Multi-Head Attention Tốt Hơn Single-Head?](#14-tại-sao-multi-head-attention-tốt-hơn-single-head)
15. [Tại Sao RCNN Fusion Layer Hoạt Động Hiệu Quả?](#15-tại-sao-rcnn-fusion-layer-hoạt-động-hiệu-quả)
16. [Tại Sao Class-Weighted Loss Có Công Thức Đó?](#16-tại-sao-class-weighted-loss-có-công-thức-đó)
17. [Tại Sao Mean Pooling/Max Pooling/Attention Pooling?](#17-tại-sao-mean-poolingmax-poolingattention-pooling)
18. [Tại Sao Dropout Hoạt Động Như Regularizer?](#18-tại-sao-dropout-hoạt-động-như-regularizer)

---

## 1. Orthogonal Initialization — Tại Sao Ổn Định Gradient?

### Vấn đề cơ bản: Ma trận lũy thừa

Trong RNN, hidden state được cập nhật qua công thức tổng quát:

```
h_t = f(W_hh · h_{t-1} + W_ih · x_t)
```

Khi lan truyền ngược qua T bước thời gian, gradient liên quan đến **lũy thừa của ma trận W_hh**:

```
∂h_T / ∂h_0 ≈ (W_hh)^T
```

### Điều gì xảy ra với (W_hh)^T?

Giả sử W_hh có **trị riêng (eigenvalues)** λ₁, λ₂, ..., λₙ. Khi lũy thừa ma trận:

```
(W_hh)^T có trị riêng: λ₁^T, λ₂^T, ..., λₙ^T
```

**Trường hợp 1: Khởi tạo ngẫu nhiên thông thường**

Nếu W_hh được khởi tạo từ phân phối chuẩn N(0, σ²) với σ = 0.01 (nhỏ):
- Các trị riêng có xu hướng |λ| < 1
- Khi T lớn: |λ|^T → 0 rất nhanh
- **Kết quả: Vanishing Gradient** — gradient biến mất trước khi đến đầu chuỗi

Nếu W_hh được khởi tạo với σ = 1.0 (lớn):
- Các trị riêng có xu hướng |λ| > 1
- Khi T lớn: |λ|^T → ∞ rất nhanh
- **Kết quả: Exploding Gradient** — gradient bùng nổ

**Trường hợp 2: Orthogonal Initialization**

Ma trận Q được gọi là **orthogonal** nếu:

```
Q^T · Q = I  (ma trận đơn vị)
```

Điều này có nghĩa:
- **Tất cả trị riêng đều có |λ| = 1**
- **Tất cả singular values đều bằng 1**
- **Q bảo toàn norm của vector: ||Qx|| = ||x||**

Khi lũy thừa ma trận orthogonal:

```
||Q^T · x|| = ||x||    với mọi T!
```

**Gradient KHÔNG bị co lại (vanish) cũng KHÔNG bị phình ra (explode)** — nó được giữ nguyên độ lớn qua mọi bước thời gian.

### Chứng minh chi tiết:

Với ma trận orthogonal Q, ta có Q^T · Q = I.

**Bảo toàn norm:**
```
||Qx||² = (Qx)^T · (Qx)
        = x^T · Q^T · Q · x
        = x^T · I · x
        = x^T · x
        = ||x||²
```

**Lũy thừa vẫn orthogonal:**
```
(Q^T)^T · (Q^T) = Q · Q^T = I    (vì Q^(-1) = Q^T)
```

Do đó Q^T cũng là ma trận orthogonal, và ||Q^T · x|| = ||x||.

### Tại sao trong code dùng `nn.init.orthogonal_`?

```python
nn.init.orthogonal_(self.W_ih)
nn.init.orthogonal_(self.W_hh)
```

PyTorch tạo ma trận orthogonal bằng **QR decomposition** của ma trận ngẫu nhiên:

```
1. Tạo ma trận ngẫu nhiên A ~ N(0, 1)
2. Phân tích QR: A = Q · R
3. Trả về Q (ma trận orthogonal)
```

### Minh họa bằng số:

Giả sử W_hh là ma trận 2×2:

**Khởi tạo ngẫu nhiên (σ=0.1):**
```
W = [[0.05, -0.08],
     [0.12,  0.03]]

Trị riêng: λ₁ ≈ 0.04 + 0.10i, λ₂ ≈ 0.04 - 0.10i
|λ| ≈ 0.11

Sau 10 bước: |λ|^10 ≈ 0.11^10 ≈ 2.6 × 10^(-10)  → GẦN NHƯ BẰNG 0!
```

**Orthogonal:**
```
Q = [[0.6,  -0.8],
     [0.8,   0.6]]

Trị riêng: λ₁ = 0.6 + 0.8i, λ₂ = 0.6 - 0.8i
|λ| = √(0.36 + 0.64) = 1.0

Sau 10 bước: |λ|^10 = 1.0^10 = 1.0  → GIỮ NGUYÊN!
Sau 1000 bước: |λ|^1000 = 1.0  → VẪN GIỮ NGUYÊN!
```

### Giới hạn của Orthogonal Initialization:

Orthogonal initialization chỉ giải quyết vấn đề **tuyến tính**. Trong thực tế, RNN có các hàm phi tuyến (sigmoid, tanh) cũng gây vanishing gradient:

```
σ'(x) = σ(x) · (1 - σ(x)) ≤ 0.25    (đạo hàm sigmoid luôn ≤ 0.25)
tanh'(x) = 1 - tanh²(x) ≤ 1.0       (đạo hàm tanh ≤ 1.0)
```

Khi lan truyền ngược qua nhiều lớp phi tuyến, tích các đạo hàm nhỏ vẫn gây vanishing. Đó là lý do LSTM cần thêm **forget gate** (xem phần 2).

---

## 2. Tại Sao LSTM Giải Quyết Được Vanishing Gradient?

### Vấn đề của RNN thông thường

Trong RNN chuẩn:

```
h_t = tanh(W · h_{t-1} + U · x_t)
```

Gradient từ h_T về h_0:

```
∂h_T / ∂h_0 = Π_{t=1}^{T} [diag(tanh'(·)) · W]
```

Mỗi thành phần `tanh'(·) ≤ 1`, và thường << 1 khi giá trị lớn. Tích T số nhỏ → 0 rất nhanh.

### LSTM thay đổi điều gì?

LSTM tạo ra một **"cao tốc gradient" (gradient highway)** qua cell state:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

Gradient của c_T theo c_0:

```
∂c_T / ∂c_0 = Π_{t=1}^{T} [diag(f_t) + (các thành phần khác)]
```

**Điểm then chốt:** Không có ma trận trọng số W nhân liên tiếp! Thay vào đó là **tích của các forget gate values f_t**.

### Tại sao forget gate giúp ích?

Forget gate: `f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)`

Giá trị f_t ∈ (0, 1). Quan trọng:

1. **Nếu mô hình học được f_t ≈ 1:** Gradient chảy qua gần như không suy giảm
   ```
   ∂c_t / ∂c_{t-1} ≈ 1.0
   ```

2. **Bias khởi tạo:** Trong nhiều cài đặt, bias của forget gate được khởi tạo với giá trị dương lớn (ví dụ b_f = 1.0), khiến f_t ≈ σ(1.0) ≈ 0.73 ngay từ đầu → gradient ban đầu đã khá ổn định.

3. **Mô hình TỰ HỌC khi nào cần nhớ/quên:** Không phải lập trình sẵn, mà mô hình học được f_t phù hợp cho từng vị trí thời gian.

### So sánh trực quan:

**RNN thường — Gradient như nước chảy qua nhiều đập:**
```
h_0 --[×0.3]--> h_1 --[×0.2]--> h_2 --[×0.4]--> h_3
Gradient: 0.3 × 0.2 × 0.4 = 0.024  (giảm 40× sau 3 bước!)
```

**LSTM — Gradient như có van điều chỉnh:**
```
c_0 --[f₁=0.9]--> c_1 --[f₂=0.95]--> c_2 --[f₃=0.85]--> c_3
Gradient: 0.9 × 0.95 × 0.85 = 0.727  (chỉ giảm 1.4× sau 3 bước!)
```

### Tại sao trong notebook không set bias forget gate lớn?

Trong `OwnLSTMCell`:

```python
nn.init.zeros_(self.b_ih)    # Bias = 0, không phải giá trị lớn
nn.init.zeros_(self.b_hh)
```

Đây là một điểm có thể cải thiện. Các cài đặt LSTM hiện đại (như trong PyTorch's nn.LSTM) thường khởi tạo bias forget gate = 1.0 để mô hình bắt đầu với xu hướng "nhớ nhiều hơn quên". Tuy nhiên, với orthogonal initialization cho weights, gradient vẫn đủ ổn định để huấn luyện.

---

## 3. Tại Sao GRU Đơn Giản Hơn Nhưng Vẫn Hiệu Quả?

### So sánh số cổng:

| | LSTM | GRU |
|--|------|-----|
| Cổng | 4 (i, f, g, o) | 2 (z, r) + 1 candidate |
| Trạng thái | 2 (h, c) | 1 (h) |
| Tham số | 4 × (d² + d·d_x + d) | 3 × (d² + d·d_x + d) |

GRU có **ít hơn 25% tham số** so với LSTM.

### GRU "nén" LSTM như thế nào?

**GRU gộp cell state và hidden state thành một:**
- LSTM: `c_t` (bộ nhớ dài hạn) và `h_t` (đầu ra ngắn hạn) tách biệt
- GRU: Chỉ có `h_t`, đóng cả hai vai trò

**GRU gộp forget gate và input gate thành update gate:**

LSTM:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t    # f_t và i_t độc lập
```

GRU:
```
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}    # z_t đóng cả 2 vai trò
```

Khi `z_t ≈ 1`: h_t ≈ h_{t-1} (giống f_t ≈ 1, i_t ≈ 0 trong LSTM — NHỚ)
Khi `z_t ≈ 0`: h_t ≈ n_t (giống f_t ≈ 0, i_t ≈ 1 trong LSTM — QUÊN và CẬP NHẬT)

**GRU gộp input gate và output gate:**
- LSTM có output gate `o_t` kiểm soát thông tin từ c_t → h_t
- GRU không có output gate riêng — reset gate `r_t` kiểm soát thông tin từ h_{t-1} → candidate

### Tại sao GRU vẫn hiệu quả?

1. **Ít tham số hơn → ít overfitting hơn** với dữ liệu nhỏ
2. **Update gate z_t** học được sự cân bằng tối ưu giữa nhớ và quên mà không cần 2 cổng riêng biệt
3. **Reset gate r_t** cho phép "xóa" thông tin cũ khi cần tạo candidate mới — tương tự forget gate

### Khi nào LSTM tốt hơn GRU?

- **Dữ liệu rất lớn:** LSTM có nhiều tham số hơn → học được biểu diễn phong phú hơn
- **Phụ thuộc rất dài hạn:** Cell state tách biệt của LSTM có thể "bảo vệ" thông tin tốt hơn
- **Trong notebook:** LSTM đạt AUC 0.9555 vs GRU 0.9441 — LSTM nhỉnh hơn một chút

---

## 4. Tại Sao Self-Attention Dùng tanh + Linear (Additive Attention)?

### Trong notebook:

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, mask=None):
        energy = torch.tanh(self.projection(hidden_states))  # tanh!
        weights = self.v(energy)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * hidden_states, dim=1)
```

### Tại sao có tanh?

Đây là **Additive Attention** (còn gọi là Bahdanau Attention). Công thức đầy đủ:

```
e_t = v^T · tanh(W · h_t + b)
α_t = softmax(e_t)
context = Σ α_t · h_t
```

**tanh hoạt động như một hàm phi tuyến** giúp mô hình học được các tương tác phức tạp giữa các chiều của hidden state.

### So sánh với các loại attention:

**1. Additive Attention (Bahdanau):**
```
score(h, s) = v^T · tanh(W_h · h + W_s · s)
```
- Phi tuyến (tanh) → biểu diễn mạnh hơn
- Tham số: O(d²)
- Chậm hơn (không vectorize tốt bằng dot-product)

**2. Dot-Product Attention (Luong):**
```
score(h, s) = h^T · s
```
- Tuyến tính → nhanh
- Không có tham số học được
- Yêu cầu d_h = d_s

**3. Scaled Dot-Product (Transformer):**
```
score(Q, K) = (Q · K^T) / √d_k
```
- Nhanh nhất (ma trận multiplication)
- Có scaling factor

### Tại sao notebook dùng Additive Attention cho RNN?

1. **Hidden states của RNN có thể có tương quan cao** giữa các chiều → tanh giúp "phi tuyến hóa" và tách biệt các patterns
2. **Chỉ có 1 hidden_size → 1 hidden_size projection** — không quá nhiều tham số thêm
3. **Đã được chứng minh hiệu quả** trong các mô hình seq2seq cổ điển (Bahdanau et al., 2014)

### Tại sao v không có bias?

```python
self.v = nn.Linear(hidden_size, 1, bias=False)
```

Vì bias trong `v` sẽ bị **triệt tiêu bởi softmax**:

```
weights = softmax(v^T · energy + b)
        = exp(v^T · energy + b) / Σ exp(v^T · energy_j + b)
        = exp(v^T · energy) · exp(b) / Σ [exp(v^T · energy_j) · exp(b)]
        = exp(v^T · energy) / Σ exp(v^T · energy_j)    # exp(b) triệt tiêu!
```

Bias không ảnh hưởng đến kết quả → không cần thiết → tiết kiệm tham số.

---

## 5. Tại Sao Scaled Dot-Product Chia Cho √d_k?

### Trong notebook:

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
```

Với `d_k = 16` (128 / 8 heads).

### Vấn đề: Phương sai của dot product

Giả sử Q và K có các phần tử độc lập, kỳ vọng 0, phương sai 1:

```
q · k = Σ_{i=1}^{d_k} q_i · k_i
```

Kỳ vọng:
```
E[q · k] = Σ E[q_i] · E[k_i] = 0
```

Phương sai:
```
Var(q · k) = Σ Var(q_i · k_i)
           = Σ E[q_i²] · E[k_i²]    (do độc lập)
           = Σ 1 · 1
           = d_k
```

**Độ lệch chuẩn = √d_k**

### Điều gì xảy ra nếu KHÔNG chia?

Với d_k = 16: độ lệch chuẩn = 4
Với d_k = 64: độ lệch chuẩn = 8
Với d_k = 256: độ lệch chuẩn = 16

Các giá trị scores có phân phối rộng → sau softmax:

```
softmax(x)_i = exp(x_i) / Σ exp(x_j)
```

Khi x_j rất lớn hoặc rất nhỏ:
- exp(lớn) → cực lớn
- exp(nhỏ) → gần 0

**Softmax bị đẩy vào vùng bão hòa:**

```
Ví dụ với d_k = 64, scores ≈ N(0, 64):
scores = [12.3, -8.7, 15.1, -3.2, ...]

exp(scores) = [220000, 0.0002, 3600000, 0.04, ...]

softmax ≈ [0.00006, ~0, 0.001, ~0, ...]
```

Gần như **one-hot distribution** — gradient của softmax gần bằng 0!

### Tại sao chia √d_k giải quyết vấn đề?

Sau khi chia:

```
scores_scaled = (q · k) / √d_k

Var(scores_scaled) = Var(q · k) / d_k = d_k / d_k = 1
```

**Phương sai = 1** → scores có phân phối chuẩn hóa → softmax không bị bão hòa → gradient ổn định.

### Minh họa bằng số:

```
d_k = 64, KHÔNG chia:
  scores ~ N(0, 64)
  softmax gradient ≈ 0 (bão hòa)

d_k = 64, CHIA √64 = 8:
  scores ~ N(0, 1)
  softmax gradient ≈ 0.2 (vùng hoạt động tốt)
```

### Tại sao không chia d_k (không căn)?

Nếu chia d_k thay vì √d_k:
```
Var = d_k / d_k² = 1/d_k
```
Với d_k = 64: Var = 1/64 ≈ 0.016 → scores quá nhỏ → softmax gần uniform → mất khả năng phân biệt.

**√d_k là điểm cân bằng:** vừa đủ để chuẩn hóa phương sai về 1, không quá nhỏ cũng không quá lớn.

---

## 6. Tại Sao LayerNorm Thay Vì BatchNorm Cho Transformer?

### Trong notebook:

```python
self.norm1 = nn.LayerNorm(d_model)
self.norm2 = nn.LayerNorm(d_model)
```

### Sự khác biệt cơ bản:

**BatchNorm:** Chuẩn hóa theo **chiều batch**
```
x_normalized = (x - μ_batch) / σ_batch
μ_batch = mean over all samples in batch, for each feature
```

**LayerNorm:** Chuẩn hóa theo **chiều features**
```
x_normalized = (x - μ_layer) / σ_layer
μ_layer = mean over all features, for each sample
```

### Tại sao BatchNorm không phù hợp với NLP/Transformer?

**1. Độ dài chuỗi thay đổi:**

Trong NLP, các câu có độ dài khác nhau. BatchNorm cần thống kê ổn định qua batch, nhưng với padding và độ dài biến đổi, thống kê này không ổn định.

```
Batch 1: ["I love cats", "The weather is nice today", "Hi"]
         → 3 tokens, 6 tokens, 2 tokens (sau padding: 6, 6, 6)

Mean của BatchNorm sẽ bị ảnh hưởng bởi padding tokens → sai lệch!
```

**2. Batch size nhỏ:**

BatchNorm cần batch size lớn (≥ 32) để thống kê chính xác. Với NLP, batch size thường nhỏ do giới hạn bộ nhớ.

```
Với batch_size = 32:
  μ và σ ước lượng từ 32 mẫu → phương sai cao → không ổn định
```

**3. Sequence dependency:**

Trong RNN/Transformer, mỗi vị trí thời gian có phân phối khác nhau (từ đầu câu khác từ cuối câu). BatchNorm áp dụng cùng μ, σ cho tất cả vị trí → không phù hợp.

### Tại sao LayerNorm phù hợp?

**1. Độc lập với batch size:**
```
LayerNorm tính μ, σ cho MỖI mẫu riêng biệt
→ Không phụ thuộc vào các mẫu khác trong batch
→ Hoạt động tốt với batch_size = 1!
```

**2. Phù hợp với sequence:**
```
LayerNorm chuẩn hóa tất cả features của một token
→ Mỗi token được chuẩn hóa độc lập
→ Không bị ảnh hưởng bởi padding
```

**3. Ổn định qua các độ dài chuỗi:**
```
Cho dù câu dài 5 tokens hay 128 tokens:
  Mỗi token đều được chuẩn hóa riêng
  → Thống kê luôn ổn định
```

### Công thức LayerNorm:

```
μ = (1/d) · Σ_{i=1}^{d} x_i
σ² = (1/d) · Σ_{i=1}^{d} (x_i - μ)²
x_normalized = (x - μ) / √(σ² + ε)    # ε = 1e-5 để tránh chia 0
x_output = γ · x_normalized + β       # γ, β là tham số học được
```

### Tại sao có γ và β (learnable parameters)?

LayerNorm không chỉ chuẩn hóa — nó còn cho phép mô hình **học lại scale và shift** phù hợp:

```
γ (weight): Học scale tối ưu cho mỗi feature
β (bias): Học shift tối ưu cho mỗi feature
```

Nếu mô hình không cần chuẩn hóa, nó có thể học γ = σ và β = μ để khôi phục giá trị gốc.

---

## 7. Tại Sao Residual Connections Giúp Gradient Flow Tốt?

### Trong notebook:

```python
attn_out = self.attention(x, x, x, mask=mask)
x = self.norm1(x + self.dropout(attn_out))    # x + ...
ff_out = self.ff(x)
x = self.norm2(x + self.dropout(ff_out))      # x + ...
```

### Vấn đề: Gradient trong mạng sâu

Xét mạng sâu L layers:

```
x_{l+1} = f_l(x_l)    với f_l là transformation của layer l
```

Gradient từ layer L về layer 0:

```
∂x_L / ∂x_0 = (∂x_L / ∂x_{L-1}) · (∂x_{L-1} / ∂x_{L-2}) · ... · (∂x_1 / ∂x_0)
            = Π_{l=0}^{L-1} J_l    (J_l là Jacobian của layer l)
```

Nếu ||J_l|| < 1 (thường xảy ra với sigmoid/tanh): tích → 0 (vanishing)
Nếu ||J_l|| > 1: tích → ∞ (exploding)

### Residual Connection thay đổi điều gì?

```
x_{l+1} = x_l + F_l(x_l)    (F_l là residual function)
```

Gradient:

```
∂x_{l+1} / ∂x_l = I + ∂F_l / ∂x_l
```

Khi lan truyền ngược qua nhiều layers:

```
∂x_L / ∂x_0 = Π_{l=0}^{L-1} (I + ∂F_l / ∂x_l)
```

Khai triển (xấp xỉ):

```
∂x_L / ∂x_0 ≈ I + Σ_{l=0}^{L-1} (∂F_l / ∂x_l) + (các số hạng bậc cao)
```

**Điểm then chốt:** Có thành phần **I (ma trận đơn vị)** trong tổng!

Ngay cả khi tất cả ∂F_l/∂x_l → 0 (các weights nhỏ ở đầu huấn luyện):

```
∂x_L / ∂x_0 ≈ I    → Gradient = 1, KHÔNG BỊ VANISH!
```

### Minh họa bằng số:

**Không có residual:**
```
Layer 1: ||J₁|| = 0.5
Layer 2: ||J₂|| = 0.5
Layer 3: ||J₃|| = 0.5

||∂x₃/∂x₀|| = 0.5 × 0.5 × 0.5 = 0.125  (giảm 8×)
```

**Có residual:**
```
Layer 1: ||I + J₁|| = ||I + 0.5|| = 1.5
Layer 2: ||I + J₂|| = 1.5
Layer 3: ||I + J₃|| = 1.5

||∂x₃/∂x₀|| ≈ 1.5 × 1.5 × 1.5 = 3.375  (KHÔNG giảm!)
```

### Tại sao residual connection hoạt động tốt với Transformer?

1. **Transformer rất sâu** (12-96 layers trong các mô hình lớn) → cần residual để gradient chảy qua
2. **Self-attention có thể có gradient nhỏ** khi attention weights phân bố đều (uniform) → residual đảm bảo gradient không biến mất
3. **Cho phép huấn luyện learning rate cao** — gradient ổn định → có thể dùng lr lớn hơn → hội tụ nhanh hơn

---

## 8. Tại Sao AdamW Tách Rời Weight Decay?

### Trong notebook:

```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### Adam thông thường (với L2 regularization):

```
g_t = ∇L(θ_t) + λ · θ_t           # Gradient + L2 regularization
m_t = β₁ · m_{t-1} + (1-β₁) · g_t # First moment
v_t = β₂ · v_{t-1} + (1-β₂) · g_t² # Second moment
m̂_t = m_t / (1 - β₁^t)            # Bias correction
v̂_t = v_t / (1 - β₂^t)
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

**Vấn đề:** L2 regularization `λ · θ_t` bị **trộn vào gradient** và sau đó bị **chia bởi adaptive learning rate** √v̂_t.

### Tại sao đây là vấn đề?

Adaptive learning rate √v̂_t khác nhau cho mỗi tham số. Điều này có nghĩa:

```
Effective weight decay cho tham số i = λ · θ_i / (√v̂_i + ε)
```

Tham số có gradient lớn (√v̂_i lớn) → effective weight decay NHỎ
Tham số có gradient nhỏ (√v̂_i nhỏ) → effective weight decay LỚN

**Weight decay không còn đồng đều** — nó bị biến dạng bởi adaptive learning rate!

### AdamW: Decoupled Weight Decay

```
g_t = ∇L(θ_t)                     # Chỉ gradient của loss
m_t = β₁ · m_{t-1} + (1-β₁) · g_t
v_t = β₂ · v_{t-1} + (1-β₂) · g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε) - η · λ · θ_t    # Weight decay TÁCH RỜI!
```

**Weight decay được áp dụng TRỰC TIẾP** sau bước Adam, không bị ảnh hưởng bởi adaptive learning rate.

### Minh họa bằng số:

Giả sử λ = 0.01, η = 0.001:

**Adam (L2 gộp):**
```
Tham số A: √v̂_A = 0.1 → effective decay = 0.01 / (0.1 + ε) ≈ 0.1
Tham số B: √v̂_B = 1.0 → effective decay = 0.01 / (1.0 + ε) ≈ 0.01
Tham số C: √v̂_C = 10.0 → effective decay = 0.01 / (10.0 + ε) ≈ 0.001

→ Weight decay khác nhau 100× giữa A và C!
```

**AdamW (tách rời):**
```
Tham số A: effective decay = 0.01
Tham số B: effective decay = 0.01
Tham số C: effective decay = 0.01

→ Weight decay ĐỒNG ĐỀU cho tất cả!
```

### Tại sao điều này quan trọng?

1. **Weight decay có mục đích rõ ràng:** Giảm độ phức tạp của mô hình, ngăn overfitting
2. **Cần áp dụng đồng đều:** Tất cả tham số nên bị "phạt" như nhau cho độ lớn
3. **Adam's adaptive lr có mục đích khác:** Điều chỉnh step size dựa trên tần suất cập nhật

Trộn hai mục đích này → cả hai đều hoạt động không tối ưu.

### Bằng chứng thực nghiệm:

Trong paper "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017):
- AdamW vượt trội hơn Adam + L2 trên ImageNet, CIFAR, và các tác vụ NLP
- Đặc biệt hiệu quả với learning rate scheduling và warmup

---

## 9. Tại Sao Warmup Giúp Huấn Luyện Ổn Định?

### Trong notebook:

```python
# Giai đoạn warmup (10% tổng steps)
if step < warmup_steps:
    lr = base_lr × (step / warmup_steps)    # Tăng tuyến tính từ 0 → base_lr
```

### Vấn đề của Adam ở đầu huấn luyện:

Adam ước lượng **first moment (m)** và **second moment (v)**:

```
m_t = β₁ · m_{t-1} + (1-β₁) · g_t
v_t = β₂ · v_{t-1} + (1-β₂) · g_t²
```

Với β₁ = 0.9, β₂ = 0.999 (mặc định):

**Ở bước đầu tiên (t=1):**
```
m₁ = 0.9 × 0 + 0.1 × g₁ = 0.1 · g₁      # Ước lượng rất tệ!
v₁ = 0.999 × 0 + 0.001 × g₁² = 0.001 · g₁²  # Ước lượng rất tệ!
```

**Bias correction:**
```
m̂₁ = m₁ / (1 - 0.9¹) = 0.1·g₁ / 0.1 = g₁     # OK
v̂₁ = v₁ / (1 - 0.999¹) = 0.001·g₁² / 0.001 = g₁²  # OK
```

Nhưng **phương sai của ước lượng rất cao** vì chỉ dựa vào 1 gradient!

### Tại sao learning rate lớn ở đầu gây vấn đề?

```
θ₂ = θ₁ - η · m̂₁ / (√v̂₁ + ε)
   = θ₁ - η · g₁ / (|g₁| + ε)
   ≈ θ₁ - η · sign(g₁)    (nếu |g₁| >> ε)
```

Với η = 0.001 (lớn): bước cập nhật có thể quá mạnh, đẩy tham số ra xa optimum.

### Warmup giải quyết như thế nào?

```
Bước 1:  lr = 0.000 → step = 0 (không cập nhật)
Bước 10: lr = 0.0001 → step nhỏ, an toàn
Bước 50: lr = 0.0005 → step trung bình
Bước 100: lr = 0.001 → step bình thường
```

**Trong giai đoạn warmup:**
1. Adam có thời gian tích lũy đủ gradient để ước lượng m và v chính xác hơn
2. Các bước cập nhật nhỏ → tham số không bị đẩy quá xa
3. Mô hình "khởi động" từ vùng tham số ngẫu nhiên → cần di chuyển cẩn thận

### Minh họa bằng hình ảnh:

```
Loss Landscape (2D):

          /\
         /  \
        /    \    ← Optimum
       /      \
      /        \
     /          \
    /            \
   S (start)------→

KHÔNG warmup (lr lớn ngay):
  S ----→ nhảy qua optimum → loss tăng → dao động

CÓ warmup (lr nhỏ → lớn):
  S -→--→---→----→ di chuyển mượt mà đến optimum
```

### Tại sao 10% warmup ratio?

Đây là heuristic phổ biến từ paper "Attention Is All You Need":

```
warmup_steps = 0.1 × total_steps
```

- **Quá ít (< 5%):** Chưa đủ thời gian để moment estimates ổn định
- **Quá nhiều (> 20%):** Lãng phí thời gian huấn luyện với lr nhỏ
- **10%:** Điểm cân bằng — đủ để ổn định, không quá dài

---

## 10. Tại Sao Cosine Decay Tốt Hơn Step Decay?

### Trong notebook:

```python
# Giai đoạn cosine decay (90% còn lại)
progress = (step - warmup_steps) / (total_steps - warmup_steps)
cosine_decay = 0.5 × (1 + cos(π × progress))
lr = base_lr × cosine_decay
```

### So sánh các strategies:

**Step Decay:**
```
lr = base_lr × 0.1    khi epoch = 5
lr = base_lr × 0.01   khi epoch = 8
→ Giảm đột ngột (step function)
```

**Linear Decay:**
```
lr = base_lr × (1 - progress)
→ Giảm đều đặn nhưng góc cạnh ở cuối
```

**Cosine Decay:**
```
lr = base_lr × 0.5 × (1 + cos(π × progress))
→ Giảm mượt mà, đạo hàm liên tục
```

### Tại sao cosine decay tốt hơn?

**1. Đạo hàm liên tục:**

Cosine có đạo hàm liên tục tại mọi điểm:
```
d(lr)/dt = -base_lr × 0.5 × π × sin(π × progress)
```

Không có điểm "gãy" → gradient của optimization process mượt mà hơn.

**2. Giảm chậm ở đầu, nhanh ở giữa, chậm ở cuối:**

```
Progress:  0.0    0.25   0.5    0.75   1.0
Cosine:    1.00   0.85   0.50   0.15   0.00
Linear:    1.00   0.75   0.50   0.25   0.00

Cosine giữ lr cao hơn ở 25% đầu → khám phá tốt hơn
Cosine giảm nhanh hơn ở giữa → hội tụ nhanh hơn
Cosine giảm chậm ở cuối → tinh chỉnh chính xác hơn
```

**3. Lý do hình học:**

Gần optimum, loss landscape thường có dạng "thung lũng hẹp":

```
        |
       / \
      /   \
     /     \    ← Thung lũng hẹp
    /_______\

lr lớn: nhảy qua lại giữa 2 bên thung lũng
lr nhỏ: di chuyển dọc theo đáy thung lũng đến optimum
```

Cosine decay tự động điều chỉnh:
- **Đầu:** lr cao → thoát khỏi local minima nông
- **Giữa:** lr giảm → tiến vào vùng optimum
- **Cuối:** lr rất thấp → tinh chỉnh chính xác

### Tại sao min_lr_ratio = 0.0?

```python
return self.min_lr_ratio + (1.0 - self.min_lr_ratio) × cosine_decay
```

Với `min_lr_ratio = 0.0`: lr cuối cùng = 0
Với `min_lr_ratio = 0.1`: lr cuối cùng = 0.1 × base_lr

**Lý do dùng 0.0:** Ở cuối huấn luyện, chúng ta muốn mô hình "định vị" chính xác tại optimum — lr = 0 cho phép điều này. Tuy nhiên, một số nghiên cứu gợi ý min_lr_ratio = 0.01-0.1 có thể tốt hơn để tránh overfitting ở epoch cuối.

---

## 11. Tại Sao Gradient Clipping Hoạt Động?

### Trong notebook:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Vấn đề: Exploding Gradient trong RNN

Trong RNN, gradient có thể bùng nổ do:

```
∂L/∂W = Σ_{t=1}^{T} (∂L/∂h_T) · (∂h_T/∂h_t) · (∂h_t/∂W)
```

Nếu ||∂h_T/∂h_t|| lớn (do eigenvalues > 1), gradient tổng có thể rất lớn.

### Gradient Clipping hoạt động như thế nào?

```python
total_norm = √(Σ ||g_i||²)    # Tính norm toàn bộ gradient

if total_norm > max_norm:
    scale = max_norm / total_norm
    g_i = g_i × scale    # Scale tất cả gradient
```

### Tại sao scaling (không phải cắt cụt) tốt hơn?

**Cắt cụt (thresholding):**
```
if |g_i| > threshold: g_i = threshold · sign(g_i)
```
- Thay đổi HƯỚNG của gradient
- Có thể đẩy optimization ra khỏi hướng tối ưu

**Scaling (norm clipping):**
```
g_i = g_i × (max_norm / total_norm)
```
- Giữ nguyên HƯỚNG của gradient
- Chỉ thay đổi ĐỘ LỚN
- Direction vẫn là direction của steepest descent

### Minh họa hình học:

```
Loss Landscape:

          ← Hướng gradient gốc (quá dài)
         /
        /
       /
      S ●━━━━━━━━━━━━━━━→ Optimum

KHÔNG clip: bước quá dài → nhảy qua optimum
Clip (scaling): cùng hướng, ngắn hơn → đến gần optimum hơn
```

### Tại sao max_norm = 1.0?

Đây là giá trị phổ biến được sử dụng trong nhiều paper:
- **Quá nhỏ (< 0.1):** Gradient bị scale quá nhiều → học chậm
- **Quá lớn (> 10):** Không ngăn được exploding gradient
- **1.0:** Điểm cân bằng — đủ nhỏ để ổn định, đủ lớn để học hiệu quả

### Gradient Clipping vs Orthogonal Initialization:

| | Orthogonal Init | Gradient Clipping |
|--|-----------------|-------------------|
| Khi nào áp dụng | Đầu huấn luyện (1 lần) | Mỗi bước cập nhật |
| Mục đích | Ngăn gradient explode/vanish từ gốc | Ngăn gradient explode khi nó xảy ra |
| Phạm vi | Chỉ ảnh hưởng khởi tạo | Ảnh hưởng toàn bộ quá trình huấn luyện |
| Kết hợp | ✓ Cả hai bổ sung cho nhau | |

---

## 12. Tại Sao BCEWithLogitsLoss Ổn Định Số Học?

### Trong notebook:

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
```

### Vấn đề: Numerical Instability của Sigmoid + BCE

BCE Loss:
```
L = -[y · log(σ(x)) + (1-y) · log(1-σ(x))]
```

Với σ(x) = 1 / (1 + e^{-x}):

**Khi x rất lớn (x → +∞):**
```
σ(x) ≈ 1
log(σ(x)) ≈ log(1) = 0     ✓ OK
log(1-σ(x)) ≈ log(0) = -∞  ✗ OVERFLOW!
```

**Khi x rất nhỏ (x → -∞):**
```
σ(x) ≈ 0
log(σ(x)) ≈ log(0) = -∞    ✗ OVERFLOW!
log(1-σ(x)) ≈ log(1) = 0   ✓ OK
```

### BCEWithLogitsLoss giải quyết bằng Log-Sum-Exp Trick:

Thay vì tính σ(x) rồi log, nó biến đổi đại số:

```
log(σ(x)) = log(1 / (1 + e^{-x}))
          = -log(1 + e^{-x})
```

Với x > 0: dùng công thức trên trực tiếp
Với x < 0: dùng identity `log(σ(x)) = x - log(1 + e^x)`

```
log(1-σ(x)) = log(σ(-x))
            = -log(1 + e^x)    (nếu x > 0)
            = -x - log(1 + e^{-x})    (nếu x < 0)
```

**Hàm softplus:** `log(1 + e^x)` được tính ổn định bằng:
```
log(1 + e^x) = x + log(1 + e^{-x})    nếu x > 0
log(1 + e^x) = log(1 + e^x)           nếu x ≤ 0
```

### Kết quả:

```
BCEWithLogitsLoss(x, y) = max(x, 0) - x·y + log(1 + e^{-|x|})
```

Công thức này:
- **KHÔNG bao giờ tính σ(x)** trực tiếp
- **KHÔNG bao giờ tính log(0)**
- **Ổn định với mọi giá trị x** từ -∞ đến +∞

### Minh họa bằng số:

```
x = 100 (rất lớn):
  σ(x) = 1.0 (trong float32)
  log(1 - σ(x)) = log(0) = -inf  ← SAI!

  BCEWithLogitsLoss:
  = max(100, 0) - 100·y + log(1 + e^{-100})
  = 100 - 100·y + 0
  = 100(1-y)    ← ĐÚNG!
```

### Tại sao kết hợp với pos_weight?

```
L = -Σ [pos_weight_c · y_c · log(σ(x_c)) + (1-y_c) · log(1-σ(x_c))]
```

pos_weight được áp dụng TRƯỚC khi tính loss → vẫn ổn định số học vì công thức log-sum-exp không thay đổi.

---

## 13. Tại Sao Positional Encoding Dùng Sin/Cos?

### Trong notebook:

```python
pe[:, 0::2] = torch.sin(position × div_term)   # Chiều chẵn: sin
pe[:, 1::2] = torch.cos(position × div_term)   # Chiều lẻ: cos
div_term = exp(arange(0, d_model, 2) × (-log(10000.0) / d_model))
```

### Công thức đầy đủ:

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

### Tại sao sin/cos thay vì học được (learnable)?

**1. Có thể ngoại suy độ dài chuỗi:**

Với learnable positional encoding:
```
PE learned cho positions 0-127
→ Không biết làm gì với position 128, 129, ...
```

Với sin/cos:
```
PE được ĐỊNH NGHĨA cho mọi pos ∈ ℕ
→ Có thể xử lý chuỗi dài hơn trong inference mà không cần huấn luyện lại
```

**2. Mối quan hệ tuyến tính giữa vị trí:**

Định lý quan trọng (Vaswani et al., 2017):

Với bất kỳ offset k cố định, PE(pos+k) có thể biểu diễn như **phép biến đổi tuyến tính** của PE(pos):

```
PE(pos+k, ·) = M_k · PE(pos, ·)
```

Với M_k là ma trận (phụ thuộc vào k, không phụ thuộc vào pos).

**Chứng minh:**

Sử dụng công thức lượng giác:
```
sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
```

Đặt α = pos/10000^{2i/d_model}, β = k/10000^{2i/d_model}:

```
PE(pos+k, 2i)   = sin(α + β)
                = sin(α)cos(β) + cos(α)sin(β)
                = PE(pos, 2i) · cos(β) + PE(pos, 2i+1) · sin(β)

PE(pos+k, 2i+1) = cos(α + β)
                = cos(α)cos(β) - sin(α)sin(β)
                = PE(pos, 2i+1) · cos(β) - PE(pos, 2i) · sin(β)
```

Đây là **phép biến đổi tuyến tính** của [PE(pos, 2i), PE(pos, 2i+1)]!

### Tại sao điều này quan trọng?

Self-attention chú ý đến **relative positions** (vị trí tương đối):

```
Attention giữa token ở pos và token ở pos+k
→ Phụ thuộc vào PE(pos) và PE(pos+k)
→ Vì PE(pos+k) = M_k · PE(pos), mô hình có thể học được M_k
→ Mô hình học được "khoảng cách k" ảnh hưởng thế nào đến attention
```

### Tại sao tần số giảm theo cấp số nhân?

```
div_term = exp(i × (-log(10000) / d_model))
         = 10000^{-i/d_model}
```

Tần số của sin/cos giảm từ 1 đến 1/10000:

```
i = 0:      tần số = 1.0        (dao động nhanh)
i = d/4:    tần số = 1/100      (dao động trung bình)
i = d/2-1:  tần số = 1/10000    (dao động chậm)
```

**Ý nghĩa:**
- **Chiều tần số cao:** Phân biệt các vị trí gần nhau (pos và pos+1 khác nhau nhiều)
- **Chiều tần số thấp:** Phân biệt các vị trí xa nhau (pos và pos+100 khác nhau rõ rệt)

Mỗi cặp chiều (2i, 2i+1) có một "tần số" khác nhau → mô hình có thể nhận biết khoảng cách ở nhiều mức độ khác nhau.

### Tại sao 10000?

Đây là hyperparameter từ paper gốc. Ý nghĩa:
- Chu kỳ dài nhất: 10000 × 2π ≈ 62,832 positions
- Đủ dài cho hầu hết các tác vụ NLP (câu thường < 512 tokens)
- Con số 10000 đủ lớn để tạo sự khác biệt rõ rệt giữa các tần số

---

## 14. Tại Sao Multi-Head Attention Tốt Hơn Single-Head?

### Trong notebook:

```python
num_heads = 8
d_k = d_model / num_heads = 128 / 8 = 16
```

### Single-Head Attention:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Tất cả thông tin được "nén" vào một không gian d chiều.

### Multi-Head Attention:

```
Head₁ = Attention(QW_q₁, KW_k₁, VW_v₁)    # d_k = 16 chiều
Head₂ = Attention(QW_q₂, KW_k₂, VW_v₂)    # d_k = 16 chiều
...
Head₈ = Attention(QW_q₈, KW_k₈, VW_v₈)    # d_k = 16 chiều

Output = Concat(Head₁, ..., Head₈) · W_o
```

### Tại sao nhiều heads tốt hơn?

**1. Nhiều không gian con (subspaces):**

Mỗi head học một loại quan hệ khác nhau trong không gian con d_k chiều:

```
Head 1: Học quan hệ cú pháp (chủ ngữ → động từ)
  "The cat [sits] on the mat"
  cat → sits: attention weight cao

Head 2: Học quan hệ ngữ nghĩa (đồng nghĩa/trái nghĩa)
  "The big [large] house"
  big → large: attention weight cao

Head 3: Học quan hệ phụ thuộc xa
  "The man [who I saw yesterday] is here"
  man → is: attention weight cao (bỏ qua mệnh đề xen giữa)

Head 4: Học quan hệ với từ phủ định
  "I do [not] like this"
  like → not: attention weight cao
```

**2. Tổng số tham số tương đương:**

Single-head với d_model = 128:
```
W_q: 128 × 128 = 16,384
W_k: 128 × 128 = 16,384
W_v: 128 × 128 = 16,384
W_o: 128 × 128 = 16,384
Total: 65,536
```

Multi-head (8 heads, d_k = 16):
```
W_q: 128 × 128 = 16,384    (vẫn 128 → 128)
W_k: 128 × 128 = 16,384
W_v: 128 × 128 = 16,384
W_o: 128 × 128 = 16,384
Total: 65,536    ← BẰNG NHAU!
```

Multi-head không thêm tham số — nó **tái phân bổ** tham số vào nhiều không gian con.

**3. Regularization effect:**

Mỗi head bị giới hạn trong d_k = 16 chiều → không thể "ghi nhớ" tất cả thông tin → buộc phải học các patterns quan trọng nhất.

### Tại sao 8 heads?

Đây là giá trị phổ biến từ paper gốc. Lý do:
- **Quá ít (2-4 heads):** Không đủ diversity trong các không gian con
- **Quá nhiều (16-32 heads):** d_k quá nhỏ (128/32 = 4) → mỗi head không đủ sức biểu diễn
- **8 heads:** d_k = 16 — đủ lớn để học patterns, đủ nhỏ để có diversity

### Khi nào nên thay đổi số heads?

| Tình huống | Khuyến nghị | Lý do |
|-----------|-------------|-------|
| d_model nhỏ (64) | 4 heads | d_k = 16 vẫn ổn |
| d_model lớn (512) | 8-16 heads | Tận dụng không gian lớn hơn |
| Dữ liệu ít | Ít heads hơn | Giảm nguy cơ overfitting |
| Dữ liệu lớn | Nhiều heads hơn | Học được nhiều patterns hơn |

---

## 15. Tại Sao RCNN Fusion Layer Hoạt Động Hiệu Quả?

### Trong notebook:

```python
# Forward LSTM outputs: [batch, seq, 256]
# Embeddings: [batch, seq, 128]
# Backward LSTM outputs: [batch, seq, 256]

combined = torch.cat([fwd_out, embeds, bwd_out], dim=2)  # [batch, seq, 640]
latent = torch.tanh(self.fusion(combined))                # [batch, seq, 512]
out, _ = torch.max(latent, dim=1)                        # [batch, 512]
```

### Tại sao fusion 3 nguồn thông tin?

**Forward context (256 chiều):**
```
Chứa thông tin từ các từ TRƯỚC từ hiện tại
Ví dụ: "I [love] cats" → "love" biết về "I" (chủ ngữ)
```

**Word embedding (128 chiều):**
```
Chứa nghĩa GỐC của từ hiện tại
Ví dụ: "love" → embedding của từ "love" (tình cảm tích cực)
```

**Backward context (256 chiều):**
```
Chứa thông tin từ các từ SAU từ hiện tại
Ví dụ: "I love [cats]" → "love" biết về "cats" (tân ngữ)
```

### Tại sao concatenation thay vì addition?

**Addition:**
```
combined = fwd_out + embeds + bwd_out    # Cần cùng số chiều
→ Thông tin bị "trộn" ngay lập tức
→ Khó phân biệt nguồn gốc của từng đặc trưng
```

**Concatenation:**
```
combined = [fwd_out | embeds | bwd_out]  # 256 + 128 + 256 = 640
→ Mỗi nguồn giữ không gian riêng
→ Linear layer học cách kết hợp tối ưu
```

Linear layer `fusion(640 → 512)` học được:
- Chiều nào từ forward quan trọng
- Chiều nào từ embedding quan trọng
- Chiều nào từ backward quan trọng
- Cách kết hợp chúng tốt nhất

### Tại sao Max Pooling over time?

```python
out, _ = torch.max(latent, dim=1)    # Max qua tất cả positions
```

**Ý nghĩa:** Với mỗi chiều trong 512 chiều, chọn giá trị lớn nhất qua toàn bộ câu.

```
Ví dụ: Phát hiện từ độc hại
latent[:, position_of_toxic_word, :] có giá trị lớn ở các chiều liên quan
→ Max pooling chọn đúng các giá trị này
→ Các vị trí khác (từ không độc hại) bị loại bỏ
```

**So với Mean Pooling:**
```
Mean Pooling: Trung bình hóa tất cả positions
→ Từ quan trọng bị "pha loãng" bởi các từ không quan trọng

Max Pooling: Chọn giá trị lớn nhất
→ Từ quan trọng được giữ nguyên, từ không quan trọng bị loại
→ Phù hợp cho phát hiện "keyword" (từ khóa độc hại)
```

### Tại sao RCNN đạt AUC cao nhất (0.9663)?

1. **Fusion layer** kết hợp đầy đủ 3 nguồn thông tin → biểu diễn phong phú nhất
2. **Max pooling** chọn đặc trưng mạnh nhất → không bị pha loãng
3. **Tanh activation** trong fusion → phi tuyến, học được tương tác phức tạp giữa 3 nguồn

---

## 16. Tại Sao Class-Weighted Loss Có Công Thức Đó?

### Trong notebook:

```python
pos_counts = df[label_columns].sum()          # Số mẫu dương mỗi lớp
neg_counts = len(df) - pos_counts             # Số mẫu âm mỗi lớp
pos_weights = neg_counts / pos_counts         # Trọng số
```

### Kết quả thực tế:

| Nhãn | pos_counts | neg_counts | pos_weight |
|------|-----------|-----------|-----------|
| toxic | ~938 | ~9062 | 9.62 |
| severe_toxic | ~102 | ~9898 | 97.04 |
| obscene | ~535 | ~9465 | 17.69 |
| threat | ~20 | ~9980 | 499.00 |
| insult | ~510 | ~9490 | 18.57 |
| identity_hate | ~91 | ~9909 | 108.89 |

### Tại sao công thức `neg_counts / pos_counts`?

**Mục tiêu:** Làm cho tổng "ảnh hưởng" của lớp dương bằng tổng ảnh hưởng của lớp âm trong loss function.

Loss không weighted:
```
L = -Σ_{i ∈ negative} log(1 - p_i) - Σ_{i ∈ positive} log(p_i)
```

Nếu có 9900 negative và 100 positive:
```
Tổng ảnh hưởng của negative: 9900 × log(1 - p)
Tổng ảnh hưởng của positive: 100 × log(p)
→ Negative chiếm 99% ảnh hưởng!
→ Mô hình chỉ cần dự đoán "negative" cho tất cả → accuracy 99%
```

Loss có weighted:
```
L = -Σ_{i ∈ negative} log(1 - p_i) - Σ_{i ∈ positive} w × log(p_i)
```

Chọn w sao cho:
```
9900 × log(1 - p) ≈ 100 × w × log(p)
→ w ≈ 9900 / 100 = 99 = neg_counts / pos_counts
```

**Khi đó:** Tổng ảnh hưởng của negative ≈ tổng ảnh hưởng của positive
→ Mô hình phải học cách phân biệt cả hai lớp!

### Tại sao không dùng (1/pos_counts) hoặc (total/pos_counts)?

**1/pos_counts:**
```
w = 1/100 = 0.01
→ Quá nhỏ! Positive càng bị "phạt" ít hơn
```

**total/pos_counts:**
```
w = 10000/100 = 100
→ Gần giống neg_counts/pos_counts (= 99)
→ Nhưng neg_counts/pos_counts chính xác hơn vì cân bằng đúng negative vs positive
```

### Tại sao threat có weight = 499?

```
pos_counts = 20 (chỉ 20 mẫu threat trong 10000)
neg_counts = 9980
pos_weight = 9980/20 = 499
```

Mỗi mẫu threat "đếm" bằng 499 mẫu không-threat trong loss function!

**Hệ quả:** Mô hình rất "sợ" bỏ sót mẫu threat → recall cao (0.50-0.75) nhưng precision thấp (0.01-0.04) vì nhiều false positives.

---

## 17. Tại Sao Mean Pooling/Max Pooling/Attention Pooling?

### Mean Pooling (LSTM, BiLSTM, GRU, Transformer):

```python
pooled_out = torch.mean(out, dim=1)    # [batch, seq, d] → [batch, d]
```

**Ưu điểm:**
- Đơn giản, nhanh
- Sử dụng thông tin từ TẤT CẢ tokens
- Ổn định (trung bình ít nhạy với outlier)

**Nhược điểm:**
- Từ quan trọng bị "pha loãng" bởi từ không quan trọng
- Padding tokens (nếu không mask) làm sai kết quả

**Phù hợp khi:**
- Tất cả tokens đều đóng góp (ví dụ: phân tích cảm xúc tổng thể)
- Cần biểu diễn "trung bình" của câu

### Max Pooling (RCNN):

```python
out, _ = torch.max(latent, dim=1)    # [batch, seq, d] → [batch, d]
```

**Ưu điểm:**
- Chọn đặc trưng MẠNH NHẤT cho mỗi chiều
- Bất biến với vị trí (từ ở đâu cũng được chọn)
- Tốt cho phát hiện "keyword"

**Nhược điểm:**
- Bỏ qua thông tin từ các tokens khác
- Nhạy với noise (outlier có thể dominate)

**Phù hợp khi:**
- Chỉ một vài tokens quan trọng (ví dụ: phát hiện từ khóa độc hại)
- Vị trí không quan trọng

### Attention Pooling (AttentionLSTM, AttentionBiLSTM):

```python
context, weights = self.attention(out, mask=attention_mask)
```

**Ưu điểm:**
- Học được weights tối ưu cho từng token
- Linh hoạt: có thể tập trung vào 1 token hoặc nhiều tokens
- Có thể mask padding tokens

**Nhược điểm:**
- Thêm tham số (projection + v)
- Có thể overfit nếu dữ liệu ít

**Phù hợp khi:**
- Cần mô hình "giải thích được" (attention weights cho biết từ nào quan trọng)
- Có tokens quan trọng và không quan trọng rõ rệt

### So sánh kết quả trong notebook:

| Mô Hình | Pooling | Best AUC | F1 Macro |
|---------|---------|----------|----------|
| LSTM | Mean | 0.9555 | 0.4354 |
| AttentionLSTM | Attention | 0.9289 | 0.4285 |
| BiLSTM | Mean | 0.9505 | 0.4106 |
| AttentionBiLSTM | Attention | 0.9483 | 0.3842 |
| RCNN | Max | 0.9663 | 0.4312 |

**Nhận xét:** Mean pooling và Max pooling hoạt động tốt hơn Attention pooling trong trường hợp này. Lý do có thể:
1. Dữ liệu ít (10K mẫu) → attention dễ overfit
2. Mean/Max pooling đơn giản hơn → ít tham số hơn → tổng quát hóa tốt hơn

---

## 18. Tại Sao Dropout Hoạt Động Như Regularizer?

### Trong notebook:

```python
dropout = nn.Dropout(0.3)    # 30% neurons bị tắt
```

### Cơ chế hoạt động:

**Trong training:**
```
Với mỗi neuron, xác suất p = 0.3 bị "tắt" (giá trị = 0)
Các neuron còn lại được scale bởi 1/(1-p) = 1/0.7 ≈ 1.43
```

**Trong inference:**
```
Tất cả neurons hoạt động
KHÔNG cần scale (PyTorch đã scale trong training)
```

### Tại sao dropout ngăn overfitting?

**1. Ngăn co-adaptation:**

Không có dropout:
```
Neuron A và Neuron B "hợp tác" chặt chẽ
→ A chỉ hoạt động tốt khi B hoạt động
→ Mô hình phụ thuộc vào sự kết hợp cụ thể này
→ Overfitting!
```

Có dropout:
```
30% khả năng B bị tắt
→ A phải học cách hoạt động độc lập
→ Mỗi neuron học các features hữu ích riêng
→ Mô hình robust hơn!
```

**2. Ensemble interpretation:**

Dropout tương đương với việc huấn luyện **2^N mô hình con** (N = số neurons) và averaging chúng:

```
Mỗi forward pass = một "sub-network" khác nhau
Sau N epochs = đã huấn luyện nhiều sub-networks
Inference = averaging tất cả sub-networks
```

**3. Noise injection:**

Dropout thêm noise vào activations:
```
a_dropout = a × mask / (1-p)    mask ~ Bernoulli(1-p)
```

Noise này hoạt động như regularizer — tương tự như thêm noise vào dữ liệu đầu vào.

### Tại sao p = 0.3?

| Dropout rate | Hiệu ứng |
|-------------|----------|
| 0.0 | Không regularization |
| 0.1-0.2 | Nhẹ, phù hợp cho dữ liệu lớn |
| 0.3-0.5 | Trung bình, phổ biến nhất |
| 0.5-0.7 | Mạnh, cho dữ liệu rất ít |
| 0.8+ | Quá mạnh, mô hình không học được |

**0.3 là điểm cân bằng:**
- Đủ để ngăn co-adaptation
- Không quá mạnh để làm chậm hội tụ
- Phù hợp cho dataset ~10K samples

### Dropout ở các vị trí khác nhau:

| Vị trí | Mục đích |
|--------|----------|
| Giữa các lớp RNN | Ngăn các lớp phụ thuộc quá mức vào nhau |
| Sau embedding | Ngăn overfitting ở tầng đầu vào |
| Trong Feed-Forward | Regularization cho sub-layer |
| Trước lớp cuối cùng | Regularization trước khi dự đoán |

Mỗi vị trí dropout "bảo vệ" một khía cạnh khác nhau của mô hình.

---

*Tài liệu này giải thích sâu về lý do toán học đằng sau mỗi kỹ thuật. Các chứng minh và công thức được trình bày ở mức độ chi tiết để người đọc có thể hiểu bản chất, không chỉ cách sử dụng.*
