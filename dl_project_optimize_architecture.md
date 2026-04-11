# Tài Liệu Chi Tiết: Dự Án Phân Loại Bình Luận Độc Hại (Toxic Comment Classification)

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Kiến Trúc Các Mô Hình](#2-kiến-trúc-các-mô-hình)
   - 2.1. [LSTM Cell (Custom Implementation)](#21-lstm-cell-custom-implementation)
   - 2.2. [GRU Cell (Custom Implementation)](#22-gru-cell-custom-implementation)
   - 2.3. [Self-Attention](#23-self-attention)
   - 2.4. [Positional Encoding](#24-positional-encoding)
   - 2.5. [Multi-Head Attention](#25-multi-head-attention)
   - 2.6. [Transformer Encoder Block](#26-transformer-encoder-block)
   - 2.7. [Mô Hình OwnLSTM](#27-mô-hình-ownlstm)
   - 2.8. [Mô Hình AttentionLSTM](#28-mô-hình-attentionlstm)
   - 2.9. [Mô Hình OwnBiLSTM](#29-mô-hình-ownbilstm)
   - 2.10. [Mô Hình AttentionBiLSTM](#210-mô-hình-attentionbilstm)
   - 2.11. [Mô Hình OwnGRU](#211-mô-hình-owngru)
   - 2.12. [Mô Hình OwnRCNN](#212-mô-hình-ownrcnn)
   - 2.13. [Mô Hình OwnTransformer](#213-mô-hình-owntransformer)
3. [Các Kỹ Thuật Tối Ưu Hóa](#3-các-kỹ-thuật-tối-ưu-hóa)
   - 3.1. [Class-Weighted Loss (BCEWithLogitsLoss với pos_weight)](#31-class-weighted-loss-bcewithlogitsloss-với-pos_weight)
   - 3.2. [AdamW Optimizer](#32-adamw-optimizer)
   - 3.3. [Gradient Clipping](#33-gradient-clipping)
   - 3.4. [Warmup Cosine Learning Rate Scheduler](#34-warmup-cosine-learning-rate-scheduler)
   - 3.5. [Early Stopping](#35-early-stopping)
   - 3.6. [Dropout Regularization](#36-dropout-regularization)
   - 3.7. [Orthogonal Weight Initialization](#37-orthogonal-weight-initialization)
4. [Tiền Xử Lý Dữ Liệu](#4-tiền-xử-lý-dữ-liệu)
5. [Đánh Giá Mô Hình](#5-đánh-giá-mô-hình)
6. [Bảng So Sánh Kết Quả](#6-bảng-so-sánh-kết-quả)

---

## 1. Tổng Quan Dự Án

Đây là một dự án học sâu hoàn chỉnh cho bài toán **Phân Loại Bình Luận Độc Hại** (Toxic Comment Classification) dựa trên bộ dữ liệu Jigsaw Toxic Comment. Mục tiêu là phân loại mỗi bình luận vào **6 nhãn độc hại** khác nhau (multi-label classification):

| Nhãn | Mô Tả |
|------|--------|
| `toxic` | Bình luận độc hại nói chung |
| `severe_toxic` | Bình luận cực kỳ độc hại |
| `obscene` | Bình luận tục tĩu |
| `threat` | Bình luận đe dọa |
| `insult` | Bình luận xúc phạm |
| `identity_hate` | Bình luận thù ghét phân biệt đối tượng |

**Điểm đặc biệt của notebook:** Tất cả các mô hình (LSTM, BiLSTM, GRU, RCNN, Transformer) đều được **cài đặt từ đầu (from scratch)** bằng PyTorch, không sử dụng các lớp có sẵn như `nn.LSTM` hay `nn.Transformer`.

### Siêu Tham Số Chính

| Tham Số | Giá Trị |
|---------|---------|
| Batch Size | 32 |
| Learning Rate | 1e-3 |
| Số Epoch tối đa | 10 (Early Stopping sẽ quyết định dừng sớm) |
| Độ dài chuỗi tối đa (MAX_LEN) | 128 |
| Kích thước Embedding (EMBEDDING_DIM) | 128 |
| Kích thước Hidden (HIDDEN_SIZE) | 256 |
| Số lớp RNN/Transformer (NUM_LAYERS) | 2 |
| Dropout | 0.3 |
| Weight Decay (L2) | 1e-4 |
| Warmup Ratio | 0.1 (10% tổng số bước) |
| Early Stopping Patience | 3 |
| Gradient Clip Max Norm | 1.0 |

---

## 2. Kiến Trúc Các Mô Hình

### 2.1. LSTM Cell (Custom Implementation)

Lớp `OwnLSTMCell` là cài đặt thủ công của một ô LSTM (Long Short-Term Memory) đơn lẻ.

#### Cấu trúc tham số:

```
W_ih: [input_size, 4 * hidden_size]    # Trọng số cho đầu vào
W_hh: [hidden_size, 4 * hidden_size]   # Trọng số cho hidden state trước
b_ih: [4 * hidden_size]                # Bias cho đầu vào
b_hh: [4 * hidden_size]                # Bias cho hidden state trước
```

#### Công thức toán học:

Tại mỗi bước thời gian `t`, với đầu vào `x_t` và trạng thái trước `(h_{t-1}, c_{t-1})`:

```
gates = x_t @ W_ih + b_ih + h_{t-1} @ W_hh + b_hh
```

4 cổng được tách từ `gates` (chia thành 4 phần bằng nhau):

1. **Cổng đầu vào (Input Gate):** `i_t = σ(W_i · x_t + U_i · h_{t-1} + b_i)`
2. **Cổng quên (Forget Gate):** `f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)`
3. **Cổng ứng viên (Candidate Gate):** `g_t = tanh(W_g · x_t + U_g · h_{t-1} + b_g)`
4. **Cổng đầu ra (Output Gate):** `o_t = σ(W_o · x_t + U_o · h_{t-1} + b_o)`

Cập nhật trạng thái:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        # Cell state mới
h_t = o_t ⊙ tanh(c_t)                   # Hidden state mới
```

Trong đó:
- `σ` là hàm sigmoid (đưa giá trị về khoảng [0, 1])
- `tanh` là hàm tang hyperbolic (đưa giá trị về khoảng [-1, 1])
- `⊙` là phép nhân từng phần tử (element-wise multiplication)

#### Khởi tạo trọng số:
- `W_ih` và `W_hh`: **Orthogonal initialization** — giúp ổn định gradient trong quá trình lan truyền ngược qua nhiều bước thời gian
- `b_ih` và `b_hh`: **Zero initialization**

#### Ý nghĩa của từng cổng:
- **Forget Gate (f_t):** Quyết định bao nhiêu thông tin từ cell state cũ `c_{t-1}` sẽ bị "quên". Giá trị gần 0 = quên hoàn toàn, gần 1 = giữ lại hoàn toàn.
- **Input Gate (i_t):** Quyết định bao nhiêu thông tin mới từ `g_t` sẽ được thêm vào cell state.
- **Candidate Gate (g_t):** Tạo ra thông tin ứng viên mới để cập nhật vào cell state.
- **Output Gate (o_t):** Quyết định bao nhiêu thông tin từ cell state sẽ được đưa ra làm hidden state.

---

### 2.2. GRU Cell (Custom Implementation)

Lớp `OwnGRUCell` là cài đặt thủ công của một ô GRU (Gated Recurrent Unit) đơn lẻ. GRU là phiên bản đơn giản hóa của LSTM với chỉ 2 cổng thay vì 4.

#### Cấu trúc tham số:

```
W_ih: [input_size, 2 * hidden_size]     # Trọng số cho update + reset gates
W_hh: [hidden_size, 2 * hidden_size]    # Trọng số recurrent cho update + reset gates
b_ih: [2 * hidden_size]                 # Bias cho update + reset gates
b_hn: [2 * hidden_size]                 # Bias recurrent cho update + reset gates
W_in: [input_size, hidden_size]         # Trọng số cho candidate hidden state
W_hn: [hidden_size, hidden_size]        # Trọng số recurrent cho candidate hidden state
b_in: [hidden_size]                     # Bias cho candidate hidden state
b_hn: [hidden_size]                     # Bias recurrent cho candidate hidden state
```

#### Công thức toán học:

1. **Cổng cập nhật (Update Gate):** `z_t = σ(W_z · x_t + U_z · h_{t-1} + b_z)`
2. **Cổng đặt lại (Reset Gate):** `r_t = σ(W_r · x_t + U_r · h_{t-1} + b_r)`
3. **Hidden state ứng viên:** `n_t = tanh(W_n · x_t + r_t ⊙ (U_n · h_{t-1}) + b_n)`
4. **Hidden state mới:** `h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}`

#### So sánh LSTM vs GRU:

| Đặc điểm | LSTM | GRU |
|----------|------|-----|
| Số cổng | 4 (input, forget, candidate, output) | 2 (update, reset) |
| Trạng thái | 2 (hidden state + cell state) | 1 (chỉ hidden state) |
| Tham số | Nhiều hơn | Ít hơn (~25-33%) |
| Tốc độ | Chậm hơn | Nhanh hơn |
| Khả năng nhớ dài hạn | Tốt hơn | Tốt (nhưng kém hơn LSTM một chút) |

---

### 2.3. Self-Attention

Lớp `SelfAttention` thực hiện cơ chế tự chú ý (additive/bahdanau-style attention) để mô hình học cách "tập trung" vào các từ quan trọng trong câu.

#### Cấu trúc:

```
projection: Linear(hidden_size → hidden_size)
v: Linear(hidden_size → 1, bias=False)
```

#### Công thức:

Với ma trận hidden states `H = [h_1, h_2, ..., h_T]` (kích thước `[batch, seq_len, hidden_size]`):

```
energy = tanh(H @ W_proj + b_proj)           # [batch, seq_len, hidden_size]
weights_raw = energy @ v                      # [batch, seq_len, 1]
```

Nếu có `attention_mask` (để loại bỏ padding tokens):
```
weights_raw[mask == 0] = -1e9                 # Gán giá trị rất nhỏ cho padding
```

```
attention_weights = softmax(weights_raw, dim=1)   # [batch, seq_len, 1]
context = Σ (attention_weights_t × h_t)           # [batch, hidden_size]
```

#### Ý nghĩa:
- **energy:** Biểu diễn mức độ "quan trọng" của mỗi hidden state
- **attention_weights:** Phân phối xác suất, tổng bằng 1, cho biết từ nào đóng góp nhiều nhất
- **context:** Vector tổng hợp có trọng số, chứa thông tin tập trung vào các phần quan trọng nhất của câu

---

### 2.4. Positional Encoding

Lớp `PositionalEncoding` thêm thông tin về vị trí của từng token trong chuỗi, vì mô hình Transformer không có cơ chế tuần tự như RNN.

#### Công thức sinusoidal:

Với vị trí `pos` và chiều `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Trong đó:
- `pos`: vị trí của token trong chuỗi (0, 1, 2, ..., max_len-1)
- `i`: chỉ số chiều trong vector embedding
- `d_model`: kích thước embedding (128 trong dự án này)
- `max_len`: 5000 (độ dài chuỗi tối đa được hỗ trợ)

#### Tại sao dùng sin/cos?
- Mỗi vị trí có một encoding duy nhất
- Cho phép mô hình học được mối quan hệ tương đối giữa các vị trí (vì `PE(pos+k)` có thể biểu diễn như hàm tuyến tính của `PE(pos)`)
- Có thể ngoại suy cho độ dài chuỗi chưa từng thấy trong quá trình huấn luyện

#### Đầu ra:
```
output = embedding + positional_encoding[:, :seq_len, :]
```

---

### 2.5. Multi-Head Attention

Lớp `MultiHeadAttention` là cốt lõi của Transformer, cho phép mô hình "chú ý" đến thông tin từ nhiều không gian biểu diễn khác nhau cùng lúc.

#### Cấu trúc:

```
d_model: kích thước embedding tổng (128)
num_heads: số head (8)
d_k = d_model / num_heads = 16    # Kích thước mỗi head
W_q: Linear(d_model → d_model)    # Query projection
W_k: Linear(d_model → d_model)    # Key projection
W_v: Linear(d_model → d_model)    # Value projection
W_o: Linear(d_model → d_model)    # Output projection
```

#### Công thức Scaled Dot-Product Attention:

```
Q = (x @ W_q).reshape(batch, seq, num_heads, d_k).transpose(1, 2)
K = (x @ W_k).reshape(batch, seq, num_heads, d_k).transpose(1, 2)
V = (x @ W_v).reshape(batch, seq, num_heads, d_k).transpose(1, 2)

scores = (Q @ K^T) / √d_k          # [batch, num_heads, seq, seq]
```

Phép chia `√d_k` (scaling factor) giúp ổn định gradient — nếu không chia, tích vô hướng của các vector có phương lớn sẽ đẩy softmax vào vùng bão hòa (gradient ≈ 0).

```
if mask is not None:
    scores[mask == 0] = -1e9       # Mask padding tokens

attn = softmax(scores, dim=-1)     # Attention weights
context = attn @ V                 # Weighted sum of values
```

#### Gộp kết quả từ nhiều head:

```
context = context.transpose(1, 2).contiguous().view(batch, seq, d_model)
output = context @ W_o
```

#### Tại sao Multi-Head?
Mỗi head có thể học các loại quan hệ khác nhau:
- Head 1: Chú ý vào quan hệ cú pháp (chủ ngữ - động từ)
- Head 2: Chú ý vào quan hệ ngữ nghĩa (từ đồng nghĩa)
- Head 3: Chú ý vào các từ phủ định
- ...

---

### 2.6. Transformer Encoder Block

Lớp `TransformerEncoderBlock` kết hợp Multi-Head Attention với Feed-Forward Network, tạo thành một khối encoder hoàn chỉnh.

#### Cấu trúc:

```
attention: MultiHeadAttention(d_model, num_heads)
norm1: LayerNorm(d_model)          # Layer Normalization sau attention
norm2: LayerNorm(d_model)          # Layer Normalization sau FFN
ff: Sequential[
    Linear(d_model → feedforward_dim),   # feedforward_dim = hidden_size * 2 = 512
    ReLU(),
    Dropout(dropout),
    Linear(feedforward_dim → d_model)
]
dropout: Dropout(dropout)
```

#### Luồng xử lý (Pre-LN style):

```
# Self-Attention + Residual Connection + LayerNorm
attn_out = attention(x, x, x, mask)
x = LayerNorm(x + Dropout(attn_out))

# Feed-Forward + Residual Connection + LayerNorm
ff_out = ff(x)
x = LayerNorm(x + Dropout(ff_out))
```

#### Các thành phần quan trọng:

1. **Residual Connection (x + ...):** Giúp gradient lan truyền tốt hơn qua nhiều lớp, giảm vấn đề vanishing gradient
2. **Layer Normalization:** Chuẩn hóa đầu ra của mỗi sub-layer, giúp huấn luyện ổn định và nhanh hơn
3. **Feed-Forward Network:** Biến đổi phi tuyến, mở rộng chiều lên `feedforward_dim` (512) rồi thu nhỏ lại về `d_model` (128), giúp mô hình học các biểu diễn phức tạp hơn
4. **Dropout:** Ngăn chặn overfitting bằng cách ngẫu nhiên "tắt" một số neuron trong quá trình huấn luyện

---

### 2.7. Mô Hình OwnLSTM

Mô hình LSTM nhiều lớp (stacked LSTM) với pooling trung bình.

#### Kiến trúc:

```
Input (input_ids) [batch, seq_len]
    ↓
Embedding [vocab_size → embedding_dim=128]
    ↓
┌─────────────────────────────────┐
│ LSTM Layer 1 (OwnLSTMCell)      │
│   input: 128, hidden: 256       │
│   → dropout (nếu num_layers > 1)│
├─────────────────────────────────┤
│ LSTM Layer 2 (OwnLSTMCell)      │
│   input: 256, hidden: 256       │
└─────────────────────────────────┘
    ↓
Mean Pooling [batch, seq_len, 256] → [batch, 256]
    ↓
Dropout(0.3)
    ↓
Linear [256 → num_classes=6]
    ↓
Output logits [batch, 6]
```

#### Đặc điểm:
- **Unidirectional:** Chỉ xử lý chuỗi theo một hướng (trái → phải)
- **Mean Pooling:** Lấy trung bình tất cả hidden states thay vì chỉ dùng hidden state cuối cùng → tận dụng thông tin từ toàn bộ chuỗi
- **Số tham số huấn luyện:** ~4,829,958

---

### 2.8. Mô Hình AttentionLSTM

Tương tự OwnLSTM nhưng thay Mean Pooling bằng Self-Attention.

#### Kiến trúc:

```
Input (input_ids) [batch, seq_len]
    ↓
Embedding [vocab_size → embedding_dim=128]
    ↓
┌─────────────────────────────────┐
│ LSTM Layer 1 (OwnLSTMCell)      │
│   input: 128, hidden: 256       │
├─────────────────────────────────┤
│ LSTM Layer 2 (OwnLSTMCell)      │
│   input: 256, hidden: 256       │
└─────────────────────────────────┘
    ↓
Self-Attention(hidden_size=256)
    ↓ [context vector, attention_weights]
Dropout(0.3)
    ↓
Linear [256 → num_classes=6]
    ↓
Output logits [batch, 6]
```

#### Ưu điểm so với OwnLSTM:
- Attention cho phép mô hình **tự động học** từ nào quan trọng nhất thay vì coi tất cả từ có trọng số bằng nhau (mean pooling)
- Có thể sử dụng `attention_mask` để loại bỏ ảnh hưởng của padding tokens
- **Số tham số huấn luyện:** ~4,896,006

---

### 2.9. Mô Hình OwnBiLSTM

Mô hình BiLSTM (Bidirectional LSTM) — xử lý chuỗi theo cả hai hướng.

#### Kiến trúc:

```
Input (input_ids) [batch, seq_len]
    ↓
Embedding [vocab_size → embedding_dim=128]
    ↓
┌──────────────────────────────────────────────────────────┐
│ BiLSTM Layer 1:                                          │
│   Forward LSTM  (trái → phải): input=128, hidden=256     │
│   Backward LSTM (phải → trái): input=128, hidden=256     │
│   Concat → [batch, seq_len, 512]                         │
│   → dropout (nếu num_layers > 1)                         │
├──────────────────────────────────────────────────────────┤
│ BiLSTM Layer 2:                                          │
│   Forward LSTM  (trái → phải): input=512, hidden=256     │
│   Backward LSTM (phải → trái): input=512, hidden=256     │
│   Concat → [batch, seq_len, 512]                         │
└──────────────────────────────────────────────────────────┘
    ↓
Mean Pooling [batch, seq_len, 512] → [batch, 512]
    ↓
Dropout(0.3)
    ↓
Linear [512 → num_classes=6]
    ↓
Output logits [batch, 6]
```

#### Đặc điểm:
- **Bidirectional:** Forward LSTM đọc từ đầu đến cuối, Backward LSTM đọc từ cuối đến đầu → nắm bắt được ngữ cảnh cả trước và sau mỗi từ
- **Concatenation:** Hidden state của forward và backward được nối lại → kích thước gấp đôi (256 × 2 = 512)
- **Số tham số huấn luyện:** ~6,277,382

---

### 2.10. Mô Hình AttentionBiLSTM

Kết hợp BiLSTM với Self-Attention.

#### Kiến trúc:

```
Input (input_ids) [batch, seq_len]
    ↓
Embedding [vocab_size → embedding_dim=128]
    ↓
┌──────────────────────────────────────────────────────────┐
│ BiLSTM Layer 1: Forward + Backward → concat [512]        │
├──────────────────────────────────────────────────────────┤
│ BiLSTM Layer 2: Forward + Backward → concat [512]        │
└──────────────────────────────────────────────────────────┘
    ↓
Self-Attention(hidden_size=512)
    ↓ [context vector, attention_weights]
Dropout(0.3)
    ↓
Linear [512 → num_classes=6]
    ↓
Output logits [batch, 6]
```

#### Ưu điểm:
- Kết hợp ưu điểm của cả **bidirectional context** (ngữ cảnh 2 chiều) và **attention** (tập trung vào từ quan trọng)
- **Số tham số huấn luyện:** ~6,540,550

---

### 2.11. Mô Hình OwnGRU

Mô hình GRU nhiều lớp (stacked GRU) với pooling trung bình.

#### Kiến trúc:

```
Input (input_ids) [batch, seq_len]
    ↓
Embedding [vocab_size → embedding_dim=128]
    ↓
┌─────────────────────────────────┐
│ GRU Layer 1 (OwnGRUCell)        │
│   input: 128, hidden: 256       │
│   → dropout (nếu num_layers > 1)│
├─────────────────────────────────┤
│ GRU Layer 2 (OwnGRUCell)        │
│   input: 256, hidden: 256       │
└─────────────────────────────────┘
    ↓
Mean Pooling [batch, seq_len, 256] → [batch, 256]
    ↓
Dropout(0.3)
    ↓
Linear [256 → num_classes=6]
    ↓
Output logits [batch, 6]
```

#### Đặc điểm:
- GRU đơn giản hơn LSTM (ít tham số hơn, nhanh hơn) nhưng vẫn giữ được khả năng xử lý phụ thuộc dài hạn
- **Số tham số huấn luyện:** ~4,599,558 (ít nhất trong các mô hình RNN)

---

### 2.12. Mô Hình OwnRCNN

Mô hình RCNN (Recurrent Convolutional Neural Network) — kết hợp RNN với cơ chế fusion.

#### Kiến trúc:

```
Input (input_ids) [batch, seq_len]
    ↓
Embedding [vocab_size → embedding_dim=128]
    ↓
┌──────────────────────────────────────────────────────────┐
│ BiLSTM Layer 1: Forward + Backward → concat [512]        │
├──────────────────────────────────────────────────────────┤
│ BiLSTM Layer 2: Forward + Backward → concat [512]        │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ Fusion:                                                   │
│   [Forward LSTM outputs | Embeddings | Backward LSTM]    │
│   → concat: [256 + 128 + 256] = [640]                    │
│   → Linear(640 → 512) + tanh                             │
└──────────────────────────────────────────────────────────┘
    ↓
Max Pooling over time [batch, seq_len, 512] → [batch, 512]
    ↓
Dropout(0.3)
    ↓
Linear [512 → num_classes=6]
    ↓
Output logits [batch, 6]
```

#### Đặc điểm nổi bật:
- **Fusion Layer:** Kết hợp 3 nguồn thông tin tại mỗi vị trí thời gian:
  1. **Forward context** (thông tin từ các từ trước)
  2. **Word embedding** (nghĩa gốc của từ hiện tại)
  3. **Backward context** (thông tin từ các từ sau)
- **Max Pooling over time:** Lấy giá trị lớn nhất qua tất cả vị trí thời gian → chọn đặc trưng mạnh nhất cho mỗi chiều
- **Số tham số huấn luyện:** ~6,605,574 (nhiều nhất trong các mô hình RNN)

---

### 2.13. Mô Hình OwnTransformer

Mô hình Transformer encoder thuần túy (không dùng pre-trained model).

#### Kiến trúc:

```
Input (input_ids) [batch, seq_len]
    ↓
Embedding [vocab_size → embedding_dim=128]
    ↓
PositionalEncoding(d_model=128, max_len=5000)
    ↓
Dropout(0.3)
    ↓
┌──────────────────────────────────────────────────┐
│ Transformer Encoder Block 1:                      │
│   MultiHeadAttention(d_model=128, num_heads=8)    │
│   → LayerNorm + Residual + Dropout                │
│   FeedForward(128 → 512 → 128) + ReLU + Dropout   │
│   → LayerNorm + Residual + Dropout                │
├──────────────────────────────────────────────────┤
│ Transformer Encoder Block 2:                      │
│   MultiHeadAttention(d_model=128, num_heads=8)    │
│   → LayerNorm + Residual + Dropout                │
│   FeedForward(128 → 512 → 128) + ReLU + Dropout   │
│   → LayerNorm + Residual + Dropout                │
└──────────────────────────────────────────────────┘
    ↓
Mean Pooling [batch, seq_len, 128] → [batch, 128]
    ↓
Linear [128 → num_classes=6]
    ↓
Output logits [batch, 6]
```

#### Đặc điểm:
- **Self-Attention:** Mỗi token có thể "nhìn thấy" tất cả các token khác trong cùng một bước → nắm bắt phụ thuộc toàn cục
- **Multi-Head (8 heads):** Mỗi head học một loại quan hệ khác nhau, `d_k = 128/8 = 16`
- **Feed-Forward dimension:** 512 (gấp 4 lần d_model — theo quy chuẩn của Transformer gốc)
- **Positional Encoding:** Sinusoidal encoding
- **Số tham số huấn luyện:** ~4,304,134 (ít nhất trong tất cả các mô hình)
- **Tốc độ:** Nhanh nhất (~20.23 it/s so với ~2.4-4.5 it/s của các mô hình RNN) do tính toán song song hoàn toàn

---

## 3. Các Kỹ Thuật Tối Ưu Hóa

### 3.1. Class-Weighted Loss (BCEWithLogitsLoss với pos_weight)

#### Vấn đề:
Dữ liệu bị **mất cân bằng lớp nghiêm trọng**. Ví dụ với 10,000 mẫu:

| Nhãn | Trọng số (pos_weight) |
|------|----------------------|
| toxic | 9.62 |
| severe_toxic | 97.04 |
| obscene | 17.69 |
| threat | 499.00 |
| insult | 18.57 |
| identity_hate | 108.89 |

Nhãn `threat` có trọng số ~499x vì cực kỳ hiếm trong dữ liệu.

#### Công thức tính trọng số:

```
pos_counts = số lượng mẫu dương của mỗi lớp
neg_counts = tổng_số_mẫu - pos_counts
pos_weight = neg_counts / pos_counts
```

#### Hàm loss:

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
```

`BCEWithLogitsLoss` kết hợp `Sigmoid + BCELoss` trong một hàm duy nhất → ổn định số học hơn (numerical stability) nhờ sử dụng log-sum-exp trick.

#### Công thức BCEWithLogitsLoss có trọng số:

```
loss = -Σ [pos_weight_c × y_c × log(σ(x_c)) + (1 - y_c) × log(1 - σ(x_c))]
```

Trong đó:
- `y_c`: nhãn thực tế của lớp c (0 hoặc 1)
- `x_c`: logit dự đoán của lớp c
- `σ`: hàm sigmoid
- `pos_weight_c`: trọng số của lớp c

---

### 3.2. AdamW Optimizer

AdamW là phiên bản cải tiến của Adam với **weight decay được tách rời (decoupled weight decay)**.

#### Tham số sử dụng:

```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

#### Khác biệt so với Adam thông thường:

| | Adam (L2 regularization) | AdamW (Decoupled weight decay) |
|--|--------------------------|-------------------------------|
| Công thức cập nhật | `g_t = ∇L(θ) + λ·θ` | `θ = θ - η·λ·θ` (tách riêng) |
| Weight decay | Gộp vào gradient | Áp dụng độc lập sau bước Adam |
| Hiệu quả | Phụ thuộc vào learning rate | Ổn định hơn, tổng quát hóa tốt hơn |

#### Tại sao dùng AdamW?
- Weight decay không bị ảnh hưởng bởi adaptive learning rate của Adam
- Giúp mô hình tổng quát hóa tốt hơn (giảm overfitting)
- Đặc biệt hiệu quả với Transformer và các mô hình lớn

---

### 3.3. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Vấn đề:
Trong các mô hình RNN, gradient có thể **bùng nổ (exploding gradient)** khi lan truyền ngược qua nhiều bước thời gian, đặc biệt với chuỗi dài.

#### Cách hoạt động:
Nếu norm của gradient vượt quá `max_norm = 1.0`, toàn bộ gradient được **scale down** proportionally:

```
total_norm = √(Σ ||g_i||²)
if total_norm > max_norm:
    g_i = g_i × (max_norm / total_norm)   # Scale tất cả gradient
```

#### Ưu điểm:
- Ngăn gradient quá lớn làm cập nhật tham số quá mạnh → mô hình phân kỳ
- Không thay đổi hướng của gradient, chỉ thay đổi độ lớn
- Đặc biệt quan trọng với các mô hình RNN custom (không có cơ chế bảo vệ sẵn)

---

### 3.4. Warmup Cosine Learning Rate Scheduler

Lớp `WarmupCosineScheduler` kết hợp hai chiến lược điều chỉnh learning rate.

#### Giai đoạn 1: Warmup (10% tổng số bước)

```
lr(step) = base_lr × (step / warmup_steps)    # Tăng tuyến tính từ 0 → base_lr
```

Ví dụ: Nếu tổng 2500 bước, warmup = 250 bước:
- Bước 0: lr = 0
- Bước 125: lr = 0.5 × base_lr
- Bước 250: lr = base_lr

#### Giai đoạn 2: Cosine Decay (90% còn lại)

```
progress = (step - warmup_steps) / (total_steps - warmup_steps)
cosine_decay = 0.5 × (1 + cos(π × progress))
lr(step) = base_lr × [min_lr_ratio + (1 - min_lr_ratio) × cosine_decay]
```

Với `min_lr_ratio = 0.0`:
- Đầu giai đoạn decay: lr = base_lr
- Giữa giai đoạn decay: lr = 0.5 × base_lr
- Cuối giai đoạn decay: lr ≈ 0

#### Tại sao Warmup + Cosine?
1. **Warmup:** Ở đầu huấn luyện, tham số còn ngẫu nhiên → gradient không ổn định. Learning rate nhỏ giúp mô hình "khởi động" an toàn
2. **Cosine Decay:** Giảm learning rate mượt mà → giúp mô hình hội tụ vào minimum tốt hơn (không bị "nhảy" qua minimum)
3. **Tổng hợp:** Kết hợp ưu điểm của cả hai — khởi động an toàn + hội tụ chính xác

---

### 3.5. Early Stopping

Lớp `EarlyStopping` tự động dừng huấn luyện khi mô hình không còn cải thiện.

#### Tham số:

```python
early_stopper = EarlyStopping(patience=3, mode='min')
```

#### Cơ chế hoạt động:

```
best_score = None
counter = 0

Mỗi epoch:
    current_score = validation_loss
    
    if best_score is None:
        best_score = current_score
        save_checkpoint()
    elif current_score >= best_score + min_delta:
        # KHÔNG cải thiện
        counter += 1
        if counter >= patience:
            early_stop = True
            restore_best_checkpoint()
    else:
        # Có cải thiện
        best_score = current_score
        counter = 0
        save_checkpoint()
```

#### Ưu điểm:
- **Tiết kiệm thời gian:** Không cần huấn luyện đủ 10 epoch nếu mô hình đã hội tụ
- **Ngăn overfitting:** Dừng đúng lúc trước khi mô hình bắt đầu ghi nhớ dữ liệu huấn luyện
- **Tự động khôi phục:** Khi dừng, mô hình được restore về trạng thái tốt nhất (không phải trạng thái cuối cùng)

#### Kết quả thực tế trong notebook:

| Mô Hình | Epoch Dừng | Best Val AUC |
|---------|-----------|-------------|
| LSTM | 8 | 0.9555 |
| BiLSTM | 6 | 0.9505 |
| Attention BiLSTM | 5 | 0.9483 |
| Attention LSTM | 5 | 0.9289 |
| GRU | 6 | 0.9441 |
| RCNN | 5 | 0.9663 |
| Transformer | 6 | 0.9429 |

---

### 3.6. Dropout Regularization

Dropout được áp dụng ở nhiều vị trí trong mô hình:

```python
dropout = nn.Dropout(0.3)    # 30% neurons bị tắt ngẫu nhiên
```

#### Vị trí áp dụng:

| Vị trí | Mục đích |
|--------|----------|
| Giữa các lớp RNN | Ngăn các lớp phụ thuộc quá mức vào nhau |
| Sau embedding (Transformer) | Ngăn overfitting ở tầng đầu vào |
| Trong Feed-Forward (Transformer) | Regularization cho sub-layer |
| Sau attention/residual (Transformer) | Stabilize training |
| Trước lớp Linear cuối cùng | Regularization trước khi dự đoán |

#### Cơ chế:
- Trong quá trình **huấn luyện:** Mỗi neuron có xác suất `p = 0.3` bị "tắt" (giá trị = 0)
- Trong quá trình **inference:** Tất cả neuron hoạt động, nhưng đầu ra được scale bởi `(1 - p)`
- Buộc mô hình học các đặc trưng **redundant** — không phụ thuộc vào bất kỳ neuron cụ thể nào

---

### 3.7. Orthogonal Weight Initialization

Tất cả các ma trận trọng số trong LSTM và GRU cells đều được khởi tạo bằng **Orthogonal Initialization**:

```python
nn.init.orthogonal_(self.W_ih)
nn.init.orthogonal_(self.W_hh)
```

#### Tại sao Orthogonal?
- Ma trận orthogonal `Q` có tính chất: `Q^T × Q = I` (ma trận đơn vị)
- Giữ nguyên norm của vector khi nhân: `||Qx|| = ||x||`
- Giúp gradient **không bị biến mất (vanishing)** hoặc **bùng nổ (exploding)** khi lan truyền qua nhiều lớp/bước thời gian
- Đặc biệt quan trọng với RNN vì gradient phải lan truyền qua nhiều bước thời gian

#### So sánh với các phương pháp khác:

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| Orthogonal | Giữ norm gradient tốt nhất cho RNN | Chỉ áp dụng cho ma trận vuông |
| Xavier/Glorot | Tốt cho các mạng feed-forward | Không tối ưu cho RNN |
| He/Kaiming | Tốt cho ReLU activation | Không phù hợp với sigmoid/tanh |
| Random normal | Đơn giản | Dễ gây vanishing/exploding gradient |

---

## 4. Tiền Xử Lý Dữ Liệu

### 4.1. Tokenizer

Sử dụng **DistilBERT Tokenizer** (`distilbert-base-uncased`):

```python
processor = TextProcessor(model_name="distilbert-base-uncased", max_len=128)
```

#### Quy trình tokenize:

1. **Input:** Văn bản thô (string)
2. **Lowercase:** Chuyển thành chữ thường (uncased)
3. **WordPiece Tokenization:** Tách từ thành subword tokens
   - Ví dụ: "unhappiness" → ["un", "##happi", "##ness"]
4. **Thêm special tokens:** `[CLS]` ở đầu, `[SEP]` ở cuối
5. **Padding:** Thêm `[PAD]` tokens để đạt độ dài 128
6. **Truncation:** Cắt bớt nếu vượt quá 128 tokens
7. **Output:** `input_ids` và `attention_mask`

#### Vocabulary size: ~30,522 tokens (vocab của DistilBERT)

### 4.2. Dataset

Lớp `ToxicDataset` kế thừa từ `torch.utils.data.Dataset`:

```python
def __getitem__(self, idx):
    text = str(row['comment_text'])
    labels = torch.tensor([toxic, severe_toxic, obscene, threat, insult, identity_hate])
    encoding = processor.tokenize(text)
    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': labels
    }
```

### 4.3. Train/Val Split

```
Train: 80% (8,000 mẫu)
Validation: 20% (2,000 mẫu)
```

Sử dụng `random_split` với seed cố định để đảm bảo tính tái lập.

---

## 5. Đánh Giá Mô Hình

### 5.1. Các Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **ROC-AUC (Macro)** | Trung bình AUC của tất cả các lớp | Khả năng phân biệt tổng quát |
| **F1-Score (Macro)** | Trung bình F1 của tất cả các lớp | Cân bằng precision/recall, không bị ảnh hưởng bởi class imbalance |
| **F1-Score (Micro)** | Tính global TP/FP/FN rồi tính F1 | Bị ảnh hưởng bởi class imbalance |
| **Precision (Macro)** | Trung bình precision của tất cả các lớp | Tỷ lệ dự đoán dương thực sự là dương |
| **Recall (Macro)** | Trung bình recall của tất cả các lớp | Tỷ lệ mẫu dương thực sự được phát hiện |
| **Subset Accuracy** | Tỷ lệ mẫu có tất cả nhãn dự đoán đúng | Metric khó nhất cho multi-label |

### 5.2. Threshold

Mặc định sử dụng `threshold = 0.5`:
```
prediction = 1 if probability >= 0.5 else 0
```

### 5.3. Classification Report

Ví dụ kết quả của mô hình LSTM tốt nhất (epoch 5):

```
              precision    recall    f1-score    support
toxic            0.38      0.84      0.52        176
severe_toxic     0.17      0.95      0.29         21
obscene          0.45      0.84      0.58        112
threat           0.01      0.50      0.02          2
insult           0.34      0.88      0.49         95
identity_hate    0.11      0.89      0.19         18

micro avg        0.30      0.86      0.45        424
macro avg        0.24      0.82      0.35        424
weighted avg     0.36      0.86      0.50        424
```

**Nhận xét:**
- **Recall cao (0.82-0.89):** Mô hình phát hiện được đa số mẫu độc hại
- **Precision thấp (0.01-0.45):** Nhiều false positives — mô hình dự đoán "độc hại" quá nhiều
- Nguyên nhân: Class-weighted loss đẩy mô hình thiên về predicting positive để tránh penalty lớn khi bỏ sót mẫu dương của các lớp hiếm

### 5.4. Per-Class ROC-AUC

```
toxic:               0.9265
severe_toxic:        0.9725
obscene:             0.9401
threat:              0.9645
insult:              0.9495
identity_hate:       0.9422
MACRO AVERAGE:       0.9492
```

AUC cao cho tất cả các lớp → mô hình có khả năng **ranking** tốt (phân biệt được mẫu dương và âm), dù threshold cố định 0.5 chưa tối ưu cho precision.

---

## 6. Bảng So Sánh Kết Quả

### 6.1. Bảng Tổng Hợp

| Mô Hình | Số Tham Số | Best Val AUC | Final AUC | F1 Macro | F1 Micro | Tốc Độ (it/s) | Epoch Dừng |
|---------|-----------|-------------|-----------|----------|----------|---------------|-----------|
| **RCNN** | 6,605,574 | **0.9663** | 0.9663 | 0.4312 | 0.5972 | ~2.39 | 5 |
| **LSTM** | 4,829,958 | 0.9555 | 0.9554 | 0.4354 | 0.5484 | ~4.48 | 8 |
| **BiLSTM** | 6,277,382 | 0.9505 | 0.9505 | 0.4106 | 0.5081 | ~2.40 | 6 |
| **Attention BiLSTM** | 6,540,550 | 0.9483 | 0.9483 | 0.3842 | 0.5260 | ~2.40 | 5 |
| **GRU** | 4,599,558 | 0.9441 | 0.9441 | 0.4847 | 0.5952 | ~3.34 | 6 |
| **Transformer** | 4,304,134 | 0.9429 | 0.9426 | **0.4934** | **0.6163** | **~20.23** | 6 |
| **Attention LSTM** | 4,896,006 | 0.9289 | 0.9289 | 0.4285 | 0.5655 | ~4.43 | 5 |

### 6.2. Phân Tích

#### Về ROC-AUC:
- **RCNN đạt cao nhất (0.9663)** nhờ fusion layer kết hợp contextual information từ cả 2 chiều với word embedding gốc, cùng max pooling chọn đặc trưng mạnh nhất
- **LSTM đứng thứ 2 (0.9555)** — bất ngờ khi unidirectional lại vượt bidirectional, có thể do mean pooling hoạt động tốt hơn với dữ liệu này
- **Transformer thấp nhất (0.9429)** trong nhóm — có thể do số epoch huấn luyện chưa đủ (Transformer thường cần nhiều dữ liệu và epoch hơn để phát huy)

#### Về F1-Score:
- **Transformer đạt F1 Macro cao nhất (0.4934)** và **F1 Micro cao nhất (0.6163)** — cho thấy Transformer có precision tốt hơn các mô hình RNN
- **GRU đạt F1 Macro thứ 2 (0.4847)** — hiệu quả tốt với ít tham số nhất trong nhóm RNN

#### Về Tốc Độ:
- **Transformer nhanh nhất (~20.23 it/s)** — gấp 4-8 lần các mô hình RNN do tính toán song song hoàn toàn (không có dependency giữa các bước thời gian)
- **LSTM/Attention LSTM nhanh nhất trong nhóm RNN (~4.4 it/s)** — do chỉ có 1 chiều
- **BiLSTM/Attention BiLSTM/RCNN chậm nhất (~2.4 it/s)** — do phải xử lý 2 chiều

#### Về Số Tham Số:
- **Transformer ít nhất (4.3M)** — do embedding dimension nhỏ (128) và không có recurrent weights
- **RCNN nhiều nhất (6.6M)** — do fusion layer và bidirectional LSTM
- **GRU ít nhất trong nhóm RNN (4.6M)** — do cấu trúc đơn giản hơn LSTM

### 6.3. Khuyến Nghị

| Mục Tiêu | Mô Hình Khuyến Nghị | Lý Do |
|----------|---------------------|-------|
| **AUC cao nhất** | RCNN | Fusion + Max Pooling cho biểu diễn mạnh nhất |
| **F1-Score cao nhất** | Transformer | Precision tốt hơn, cân bằng hơn |
| **Tốc độ inference** | Transformer | ~20 it/s, song song hoàn toàn |
| **Ít tham số nhất** | Transformer | 4.3M tham số |
| **Cân bằng tất cả** | GRU | F1 tốt, tham số ít, tốc độ trung bình |

---

## Phụ Lục: Sơ Đồ Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT PIPELINE                               │
│  Raw Text → DistilBERT Tokenizer → input_ids + attention_mask       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │   RNN Models  │ │   RCNN     │ │ Transformer│
    │               │ │            │ │            │
    │ Embedding(128)│ │ Embed(128) │ │ Embed(128) │
    │               │ │            │ │ + PosEnc   │
    │ LSTM/GRU/     │ │ BiLSTM     │ │            │
    │ BiLSTM        │ │ × 2 layers │ │ 2× EncBlock│
    │ × 2 layers    │ │            │ │ (8 heads)  │
    │               │ │ Fusion     │ │            │
    │ Mean Pool /   │ │ (256+128   │ │ Mean Pool  │
    │ Attention     │ │  +256)     │ │            │
    │               │ │ Max Pool   │ │            │
    └─────────┬─────┘ └─────┬──────┘ └─────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Dropout(0.3)   │
                    │  Linear(→6)     │
                    │  BCEWithLogits  │
                    │  + pos_weight   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  AdamW          │
                    │  + weight_decay │
                    │  + grad clip    │
                    │  + LR scheduler │
                    │  + early stop   │
                    └─────────────────┘
```

---

*Tài liệu được tạo dựa trên phân tích chi tiết notebook `dl-project-optimize.ipynb`.*
