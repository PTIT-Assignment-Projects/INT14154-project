# Giải thích chi tiết LSTM Cell và LSTM Models

## Mục lục

1. [OwnLSTMCell - Custom LSTM Cell](#1-ownlstmcell---custom-lstm-cell)
2. [OwnLSTM - Unidirectional LSTM Classifier](#2-ownlstm---unidirectional-lstm-classifier)
3. [SelfAttention - Attention Mechanism](#3-selfattention---attention-mechanism)
4. [AttentionLSTM - LSTM với Attention](#4-attentionlstm---lstm-với-attention)
5. [OwnBiLSTM - Bidirectional LSTM](#5-ownbilstm---bidirectional-lstm)
6. [AttentionBiLSTM - BiLSTM với Attention](#6-attentionbilstm---bilstm-với-attention)
7. [So sánh các mô hình](#7-so-sánh-các-mô-hình)

---

## 1. OwnLSTMCell - Custom LSTM Cell

**File:** `src/models/lstm/lstm_cell.py`

### 1.1 Tổng quan về LSTM

LSTM (Long Short-Term Memory) là một kiến trúc mạng neural đặc biệt được thiết kế để xử lý vấn đề **vanishing gradient** trong RNN truyền thống. LSTM sử dụng **cơ chế gate** để kiểm soát dòng thông tin qua 3 thành phần chính:

- **Forget Gate (f):** Quyết định thông tin nào cần loại bỏ khỏi cell state
- **Input Gate (i):** Quyết định thông tin mới nào cần lưu vào cell state
- **Output Gate (o):** Quyết định thông tin nào từ cell state sẽ trở thành hidden state

### 1.2 Cấu trúc `__init__`

```python
def __init__(self, input_size, hidden_size):
    self.W_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
    self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
    self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
    self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
```

**Giải thích chi tiết:**

| Tham số | Shape | Ý nghĩa |
|---------|-------|---------|
| `W_ih` | `(input_size, 4 * hidden_size)` | Ma trận trọng số cho **input** → tính cả 4 gates cùng lúc |
| `W_hh` | `(hidden_size, 4 * hidden_size)` | Ma trận trọng số cho **hidden state trước** → tính cả 4 gates cùng lúc |
| `b_ih` | `(4 * hidden_size,)` | Bias cho input transformation |
| `b_hh` | `(4 * hidden_size,)` | Bias cho hidden state transformation |

**Tại sao `4 * hidden_size`?** Thay vì tạo 4 ma trận riêng biệt cho mỗi gate, code gộp tất cả vào một ma trận lớn để tính toán song song → **tăng hiệu suất** đáng kể trên GPU.

### 1.3 Khởi tạo tham số (`reset_parameters`)

```python
def reset_parameters(self):
    nn.init.orthogonal_(self.W_ih)
    nn.init.orthogonal_(self.W_hh)
    nn.init.zeros_(self.b_ih)
    nn.init.zeros_(self.b_hh)
```

- **Orthogonal initialization:** Giữ cho gradient không bị explode/vanish qua nhiều time step. Ma trận orthogonal bảo toàn norm của vector khi nhân → ổn định training cho sequence dài.
- **Zero initialization cho bias:** Bắt đầu từ trạng thái trung tính, không thiên vị gate nào.

### 1.4 Forward Pass

```python
def forward(self, x, states):
    h_prev, c_prev = states
    
    # Tính toán gates
    gates = (x @ W_ih + b_ih) + (h_prev @ W_hh + b_hh)
    
    # Tách thành 4 gates
    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
    
    # Áp dụng activation functions
    i_gate = torch.sigmoid(i_gate)    # Input gate
    f_gate = torch.sigmoid(f_gate)    # Forget gate
    g_gate = torch.tanh(g_gate)       # Candidate cell state
    o_gate = torch.sigmoid(o_gate)    # Output gate
    
    # Cập nhật cell state và hidden state
    c_next = f_gate * c_prev + i_gate * g_gate
    h_next = o_gate * torch.tanh(c_next)
    
    return h_next, c_next
```

**Chi tiết từng bước:**

#### Bước 1: Tính toán gates

```
gates = (x @ W_ih + b_ih) + (h_prev @ W_hh + b_hh)
```

Kết quả có shape `(batch_size, 4 * hidden_size)`, chứa giá trị cho cả 4 gates.

#### Bước 2: Tách gates

```python
i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
```

`chunk(4, 1)` chia tensor thành 4 phần bằng nhau theo chiều 1 (column).

#### Bước 3: Activation functions

| Gate | Activation | Output range | Ý nghĩa |
|------|-----------|--------------|---------|
| `i_gate` | sigmoid | [0, 1] | Tỷ lệ thông tin mới cần thêm vào |
| `f_gate` | sigmoid | [0, 1] | Tỷ lệ thông tin cũ cần giữ lại |
| `g_gate` | tanh | [-1, 1] | Giá trị ứng viên mới cho cell state |
| `o_gate` | sigmoid | [0, 1] | Tỷ lệ thông tin từ cell state để output |

#### Bước 4: Cập nhật states

**Cell state mới:**
```
c_next = f_gate * c_prev + i_gate * g_gate
```
- `f_gate * c_prev`: Giữ lại một phần cell state cũ (controlled forgetting)
- `i_gate * g_gate`: Thêm thông tin mới (controlled input)

**Hidden state mới:**
```
h_next = o_gate * tanh(c_next)
```
- `tanh(c_next)` đưa cell state về [-1, 1]
- `o_gate` kiểm soát bao nhiêu thông tin được output

### 1.5 Minh họa LSTM Cell

```
                ┌─────────────────────────────────────────┐
                │                                         │
 x ────────────►│  gates = x@W_ih + h_prev@W_hh + biases  │
                │                                         │
 h_prev ───────►│         chunk(4) → i, f, g, o           │
                │                                         │
                │  i = σ(i_gate)    f = σ(f_gate)         │
 c_prev ───────►│  g = tanh(g)    o = σ(o_gate)           │
                │                                         │
                │  c_next = f * c_prev + i * g            │──► c_next
                │  h_next = o * tanh(c_next)              │──► h_next
                │                                         │
                └─────────────────────────────────────────┘
```

---

## 2. OwnLSTM - Unidirectional LSTM Classifier

**File:** `src/models/lstm/lstm_classifier.py`

### 2.1 Kiến trúc tổng quan

```
Input (indices) 
    │
    ▼
┌──────────────┐
│  Embedding   │  (vocab_size → embedding_dim)
└──────────────┘
    │
    ▼
┌──────────────┐
│  LSTM Layer 1 │  (OwnLSTMCell × seq_len steps)
└──────────────┘
    │
    ▼
┌──────────────┐
│  Dropout     │  (giữa các layers)
└──────────────┘
    │
    ▼
┌──────────────┐     (lặp lại cho num_layers)
│  LSTM Layer N │
└──────────────┘
    │
    ▼
┌──────────────────────┐
│ Global Average Pool  │  mean over seq_len
└──────────────────────┘
    │
    ▼
┌──────────────┐
│  Dropout     │
└──────────────┘
    │
    ▼
┌──────────────┐
│  Linear (fc) │  (hidden_size → num_classes)
└──────────────┘
    │
    ▼
  Logits
```

### 2.2 Khởi tạo (`__init__`)

```python
def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        layer_input_size = embedding_dim if i == 0 else hidden_size
        self.layers.append(OwnLSTMCell(layer_input_size, hidden_size))
    
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_size, num_classes)
```

**Giải thích:**

- **Embedding:** Chuyển đổi token indices thành dense vectors
- **ModuleList:** Container cho nhiều LSTM cells (mỗi cell = 1 layer)
- **layer_input_size:** 
  - Layer đầu tiên nhận `embedding_dim` (output của embedding layer)
  - Các layer sau nhận `hidden_size` (output của LSTM layer trước)
- **fc:** Linear layer cuối cùng để phân loại

### 2.3 Forward Pass

```python
def forward(self, x, attention_mask=None):
    batch_size, seq_len = x.size()
    out = self.embedding(x)  # (batch, seq_len, embedding_dim)
    
    for layer_idx in range(self.num_layers):
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        layer_outputs = []
        for t in range(seq_len):
            h, c = self.layers[layer_idx](out[:, t, :], (h, c))
            layer_outputs.append(h)
        
        out = torch.stack(layer_outputs, dim=1)  # (batch, seq_len, hidden_size)
        
        if layer_idx < self.num_layers - 1:
            out = self.dropout(out)
    
    pooled_out = torch.mean(out, dim=1)  # Global Average Pooling
    pooled_out = self.dropout(pooled_out)
    logits = self.fc(pooled_out)
    
    return logits
```

**Chi tiết từng bước:**

#### Bước 1: Embedding

```python
out = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
```

Chuyển đổi mỗi token index thành vector dense.

#### Bước 2: Xử lý qua từng LSTM layer

```python
for layer_idx in range(self.num_layers):
    h = torch.zeros(batch_size, self.hidden_size).to(x.device)
    c = torch.zeros(batch_size, self.hidden_size).to(x.device)
```

- Mỗi layer bắt đầu với hidden state và cell state = 0
- **Lý do:** Không có thông tin trước đó, bắt đầu từ "trạng thái trống"

#### Bước 3: Unroll qua time steps

```python
for t in range(seq_len):
    h, c = self.layers[layer_idx](out[:, t, :], (h, c))
    layer_outputs.append(h)
```

- `out[:, t, :]` lấy vector embedding tại vị trí thời gian `t`
- LSTM cell tính toán và cập nhật `h`, `c`
- Lưu `h` tại mỗi time step để làm input cho layer tiếp theo

#### Bước 4: Stack outputs

```python
out = torch.stack(layer_outputs, dim=1)  # (batch, seq_len, hidden_size)
```

Ghép tất cả hidden states thành tensor 3D.

#### Bước 5: Dropout giữa các layers

```python
if layer_idx < self.num_layers - 1:
    out = self.dropout(out)
```

Chỉ áp dụng dropout giữa các layers, **không** áp dụng sau layer cuối cùng (vì đã có dropout riêng trước fc).

#### Bước 6: Global Average Pooling

```python
pooled_out = torch.mean(out, dim=1)  # (batch, hidden_size)
```

Lấy trung bình hidden states qua tất cả time steps → vector cố định bất kể độ dài sequence.

#### Bước 7: Classification

```python
pooled_out = self.dropout(pooled_out)
logits = self.fc(pooled_out)  # (batch, num_classes)
```

Áp dụng dropout rồi linear layer để dự đoán.

### 2.4 Shape transformations

| Bước | Shape |
|------|-------|
| Input `x` | `(batch_size, seq_len)` |
| After embedding | `(batch_size, seq_len, embedding_dim)` |
| After LSTM layer 1 | `(batch_size, seq_len, hidden_size)` |
| After LSTM layer N | `(batch_size, seq_len, hidden_size)` |
| After pooling | `(batch_size, hidden_size)` |
| Output logits | `(batch_size, num_classes)` |

---

## 3. SelfAttention - Attention Mechanism

**File:** `src/models/lstm/attention.py`

### 3.1 Tổng quan

Attention mechanism cho phép mô hình **học trọng số** cho từng time step thay vì coi tất cả equal (như average pooling). Time step quan trọng hơn sẽ có trọng số cao hơn.

### 3.2 Additive Attention (Bahdanau-style)

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
```

**Giải thích:**

- **`projection`:** Linear layer để transform hidden states → không gian mới để tính attention
- **`v`:** Linear layer để tính attention score → output scalar cho mỗi time step

### 3.3 Forward Pass

```python
def forward(self, hidden_states, mask=None):
    # hidden_states: (batch_size, seq_len, hidden_size)
    
    energy = torch.tanh(self.projection(hidden_states))  # (batch, seq_len, hidden_size)
    weights = self.v(energy)                              # (batch, seq_len, 1)
    
    if mask is not None:
        mask = mask.unsqueeze(-1).float()
        weights = weights.masked_fill(mask == 0, -1e9)
    
    weights = F.softmax(weights, dim=1)
    context = torch.sum(weights * hidden_states, dim=1)
    
    return context, weights
```

**Chi tiết từng bước:**

#### Bước 1: Tính energy

```python
energy = torch.tanh(self.projection(hidden_states))
```

- Transform hidden states qua linear layer + tanh
- Tạo ra "biểu diễn" phù hợp để tính attention

#### Bước 2: Tính attention scores

```python
weights = self.v(energy)  # (batch, seq_len, 1)
```

- `v` là vector query học được
- Output là scalar score cho mỗi time step

#### Bước 3: Masking (cho padding tokens)

```python
if mask is not None:
    mask = mask.unsqueeze(-1).float()
    weights = weights.masked_fill(mask == 0, -1e9)
```

- `mask`: 1 cho token thật, 0 cho padding
- `masked_fill`: Gán `-1e9` cho padding tokens → softmax sẽ cho weight ≈ 0
- **Quan trọng:** Masking trước softmax để padding không ảnh hưởng đến attention distribution

#### Bước 4: Softmax

```python
weights = F.softmax(weights, dim=1)
```

- Chuẩn hóa weights thành phân phối xác suất (tổng = 1)
- `dim=1`: softmax qua chiều sequence length

#### Bước 5: Weighted sum → Context vector

```python
context = torch.sum(weights * hidden_states, dim=1)
```

- Nhân weights với hidden states → weighted sum
- Kết quả: `context` vector đại diện cho toàn bộ sequence, tập trung vào các time step quan trọng

### 3.4 Minh họa Attention

```
hidden_states: [h1, h2, h3, ..., hT]  (batch, seq_len, hidden_size)
                     │
                     ▼
        ┌────────────────────────┐
        │   projection + tanh    │
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │      v (Linear)        │
        └────────────────────────┘
                     │
                     ▼
    scores: [s1, s2, s3, ..., sT]  (batch, seq_len, 1)
                     │
                     ▼
        ┌────────────────────────┐
        │       Softmax          │
        └────────────────────────┘
                     │
                     ▼
    weights: [α1, α2, α3, ..., αT]  (tổng = 1)
                     │
                     ▼
    context = Σ(αt * ht)  →  (batch, hidden_size)
```

---

## 4. AttentionLSTM - LSTM với Attention

**File:** `src/models/lstm/attention_lstm.py`

### 4.1 Khác biệt so với OwnLSTM

| Thành phần | OwnLSTM | AttentionLSTM |
|-----------|---------|---------------|
| Pooling | Global Average | SelfAttention |
| Input cho fc | `mean(hidden_states)` | `attention(hidden_states)` |
| Khả năng | Coi tất cả time steps equal | Học trọng số cho từng time step |

### 4.2 Kiến trúc

```
Input → Embedding → LSTM Layers → SelfAttention → Dropout → FC → Logits
```

### 4.3 Forward Pass

```python
# Giống OwnLSTM đến bước này
out = torch.stack(layer_outputs, dim=1)  # (batch, seq_len, hidden_size)

# Khác biệt: Attention pooling thay vì average pooling
context, attn_weights = self.attention(out, mask=attention_mask)

# Classification
context = self.dropout(context)
logits = self.fc(context)
```

**Lợi ích của Attention:**

1. **Tập trung vào từ khóa:** Mô hình học được từ nào quan trọng cho classification
2. **Xử lý sequence dài tốt hơn:** Không bị "loãng" thông tin như average pooling
3. **Interpretability:** Có thể visualize attention weights để hiểu mô hình "nhìn" vào đâu

---

## 5. OwnBiLSTM - Bidirectional LSTM

**File:** `src/models/lstm/bilstm/bilstm_classifier.py`

### 5.1 Tại sao cần Bidirectional?

LSTM thông thường chỉ đọc sequence **theo 1 hướng** (trái → phải). BiLSTM đọc **cả 2 hướng**:

- **Forward LSTM:** Đọc từ đầu đến cuối sequence
- **Backward LSTM:** Đọc từ cuối đến đầu sequence

→ Mô hình có context từ **cả quá khứ và tương lai** tại mỗi time step.

### 5.2 Kiến trúc

```
                    ┌─────────────────┐
Input → Embedding ──┤                 ├──► Forward outputs
                    │   Forward LSTM  │
                    │   (→ direction) │
                    └─────────────────┘
                    
                    ┌─────────────────┐
Input → Embedding ──┤                 ├──► Backward outputs
                    │  Backward LSTM  │
                    │   (← direction) │
                    └─────────────────┘
                    
                    ┌─────────────────┐
Forward + Backward ─┤   Concatenate   ├──► (batch, seq_len, hidden*2)
outputs             └─────────────────┘
                           │
                           ▼
                    Global Average Pool
                           │
                           ▼
                        FC → Logits
```

### 5.3 Khởi tạo

```python
def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
    self.fwd_layers = nn.ModuleList()
    self.bwd_layers = nn.ModuleList()
    
    for i in range(num_layers):
        layer_input_size = embedding_dim if i == 0 else hidden_size * 2
        self.fwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))
        self.bwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))
    
    self.fc = nn.Linear(hidden_size * 2, num_classes)
```

**Điểm quan trọng:**

- **`hidden_size * 2` cho layer sau:** Output của BiLSTM là concatenation của forward + backward → kích thước gấp đôi
- **`hidden_size * 2` cho fc:** Input của classification layer cũng gấp đôi

### 5.4 Forward Pass

```python
def forward(self, x, attention_mask=None):
    batch_size, seq_len = x.size()
    out = self.embedding(x)
    
    for layer_idx in range(self.num_layers):
        # Initialize states
        h_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
        h_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        fwd_outputs = []
        bwd_outputs = [None] * seq_len
        
        # Forward pass: t = 0 → seq_len-1
        for t in range(seq_len):
            h_fwd, c_fwd = self.fwd_layers[layer_idx](out[:, t, :], (h_fwd, c_fwd))
            fwd_outputs.append(h_fwd)
        
        # Backward pass: t = seq_len-1 → 0
        for t in range(seq_len - 1, -1, -1):
            h_bwd, c_bwd = self.bwd_layers[layer_idx](out[:, t, :], (h_bwd, c_bwd))
            bwd_outputs[t] = h_bwd
        
        # Stack và concatenate
        fwd_outputs = torch.stack(fwd_outputs, dim=1)
        bwd_outputs = torch.stack(bwd_outputs, dim=1)
        out = torch.cat((fwd_outputs, bwd_outputs), dim=2)
        
        if layer_idx < self.num_layers - 1:
            out = self.dropout(out)
    
    pooled_out = torch.mean(out, dim=1)
    pooled_out = self.dropout(pooled_out)
    logits = self.fc(pooled_out)
    
    return logits
```

**Chi tiết quan trọng:**

#### Forward pass

```python
for t in range(seq_len):  # 0, 1, 2, ..., seq_len-1
    h_fwd, c_fwd = self.fwd_layers[layer_idx](out[:, t, :], (h_fwd, c_fwd))
    fwd_outputs.append(h_fwd)
```

- Đọc sequence từ **trái sang phải**
- Time step `t` có context từ các time steps `0, 1, ..., t-1`

#### Backward pass

```python
for t in range(seq_len - 1, -1, -1):  # seq_len-1, seq_len-2, ..., 0
    h_bwd, c_bwd = self.bwd_layers[layer_idx](out[:, t, :], (h_bwd, c_bwd))
    bwd_outputs[t] = h_bwd
```

- Đọc sequence từ **phải sang trái**
- Time step `t` có context từ các time steps `seq_len-1, seq_len-2, ..., t+1`
- **`bwd_outputs[t] = h_bwd`:** Gán vào đúng vị trí `t` để giữ thứ tự sequence

#### Concatenate

```python
out = torch.cat((fwd_outputs, bwd_outputs), dim=2)
```

- `dim=2`: Concatenate theo chiều hidden dimension
- Kết quả: `(batch, seq_len, hidden_size * 2)`
- Tại mỗi position `t`: `[h_fwd_t | h_bwd_t]` chứa context từ cả 2 hướng

### 5.5 Shape transformations

| Bước | Shape |
|------|-------|
| Input `x` | `(batch_size, seq_len)` |
| After embedding | `(batch_size, seq_len, embedding_dim)` |
| After BiLSTM layer 1 | `(batch_size, seq_len, hidden_size * 2)` |
| After BiLSTM layer N | `(batch_size, seq_len, hidden_size * 2)` |
| After pooling | `(batch_size, hidden_size * 2)` |
| Output logits | `(batch_size, num_classes)` |

---

## 6. AttentionBiLSTM - BiLSTM với Attention

**File:** `src/models/lstm/bilstm/attention_bilstm.py`

### 6.1 Kiến trúc

Kết hợp **BiLSTM** + **Attention**:

```
Input → Embedding → BiLSTM Layers → SelfAttention → Dropout → FC → Logits
```

### 6.2 Khác biệt so với OwnBiLSTM

```python
# Attention trên concatenated output (hidden_size * 2)
self.attention = SelfAttention(hidden_size * 2)
self.fc = nn.Linear(hidden_size * 2, num_classes)
```

- Attention nhận input là `hidden_size * 2` (vì output của BiLSTM đã được concatenate)
- Thay vì average pooling, dùng attention để weighted pooling

### 6.3 Forward Pass

```python
# BiLSTM processing (giống OwnBiLSTM)
out = torch.cat((fwd_outputs, bwd_outputs), dim=2)  # (batch, seq_len, hidden*2)

# Attention pooling
context, attn_weights = self.attention(out, mask=attention_mask)

# Classification
context = self.dropout(context)
logits = self.fc(context)
```

### 6.4 Lợi ích kết hợp BiLSTM + Attention

| Thành phần | Lợi ích |
|-----------|---------|
| **BiLSTM** | Context từ cả 2 hướng (quá khứ + tương lai) |
| **Attention** | Tập trung vào time steps quan trọng, ignore noise |
| **Kết hợp** | Mô hình hiểu context đầy đủ + biết chọn lọc thông tin |

---

## 7. So sánh các mô hình

### 7.1 Bảng so sánh

| Mô hình | Hướng | Pooling | Parameters | Độ phức tạp | Accuracy tiềm năng |
|---------|-------|---------|------------|-------------|-------------------|
| **OwnLSTM** | Unidirectional | Average | Thấp nhất | Thấp nhất | Trung bình |
| **AttentionLSTM** | Unidirectional | Attention | Trung bình | Trung bình | Khá tốt |
| **OwnBiLSTM** | Bidirectional | Average | Cao | Cao | Tốt |
| **AttentionBiLSTM** | Bidirectional | Attention | Cao nhất | Cao nhất | Tốt nhất |

### 7.2 Khi nào dùng mô hình nào?

| Tình huống | Mô hình khuyến nghị |
|-----------|---------------------|
| Dataset nhỏ, cần train nhanh | OwnLSTM |
| Dataset trung bình, cần interpretability | AttentionLSTM |
| Dataset lớn, cần accuracy cao | OwnBiLSTM |
| Dataset lớn, cần accuracy + interpretability | AttentionBiLSTM |
| Sequence dài, từ khóa quan trọng | AttentionLSTM/AttentionBiLSTM |
| Context 2 chiều quan trọng | BiLSTM/AttentionBiLSTM |

### 7.3 Công thức tính số parameters

**OwnLSTMCell:**
```
Params = input_size * 4*hidden + hidden * 4*hidden + 4*hidden + 4*hidden
       = 4 * hidden * (input_size + hidden + 2)
```

**OwnLSTM (multi-layer):**
```
Embedding: vocab_size * embedding_dim
LSTM layers: num_layers * 4 * hidden * (layer_input + hidden + 2)
FC: hidden * num_classes + num_classes
```

**BiLSTM:**
```
≈ 2x số parameters của LSTM (vì có 2 directions)
FC input: hidden * 2 (do concatenate)
```

### 7.4 Minh họa kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT TEXT                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EMBEDDING LAYER                               │
│  token indices → dense vectors                                  │
│  Shape: (batch, seq_len, embedding_dim)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LSTM/BiLSTM LAYERS                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Layer 1   │───►│   Layer 2   │───►│   Layer N   │         │
│  │  + Dropout  │    │  + Dropout  │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│  Shape: (batch, seq_len, hidden_size) hoặc (hidden_size * 2)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  Average Pooling     │  │    Attention Pooling     │
│  mean(dim=1)         │  │  weighted sum            │
│  (simple, fast)      │  │  (learnable, powerful)   │
└──────────┬───────────┘  └──────────┬───────────────┘
           │                         │
           ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION HEAD                           │
│  Dropout → Linear(hidden, num_classes) → Logits                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Công thức toán học LSTM

### 8.1 Các phương trình cốt lõi

```
Forget gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:      g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
Output gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Cell state:     c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden state:   h_t = o_t ⊙ tanh(c_t)
```

Trong đó:
- `σ`: Sigmoid function
- `⊙`: Element-wise multiplication (Hadamard product)
- `[h_{t-1}, x_t]`: Concatenation của hidden state trước và input hiện tại

### 8.2 Tại sao LSTM giải quyết vanishing gradient?

Cell state `c_t` có **connection trực tiếp** qua thời gian:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

- Khi `f_t ≈ 1`, gradient flow gần như không bị suy giảm qua time steps
- Forget gate học được khi nào cần "nhớ" (f ≈ 1) và khi nào cần "quên" (f ≈ 0)
- Đây gọi là **constant error carousel (CEC)**

---

## 9. Lưu ý quan trọng

### 9.1 Thứ tự operations trong code

Code sử dụng **concatenated weight matrices** thay vì 4 ma trận riêng:

```python
# Thay vì:
W_i, W_f, W_g, W_o = ...  # 4 ma trận riêng

# Code dùng:
W_combined = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
gates = x @ W_combined  # Tính 1 lần, sau đó chunk
```

**Lợi ích:**
- Ít phép nhân ma trận hơn → nhanh hơn trên GPU
- Memory access pattern tốt hơn → tận dụng cache

### 9.2 Masking trong Attention

```python
weights = weights.masked_fill(mask == 0, -1e9)
```

- `-1e9` thay vì `-inf` để tránh numerical instability
- Sau softmax, weights cho padding tokens ≈ 0

### 9.3 Dropout placement

```python
# Giữa các LSTM layers
if layer_idx < self.num_layers - 1:
    out = self.dropout(out)

# Trước classification
pooled_out = self.dropout(pooled_out)
```

- Dropout **giữa layers**: Giảm overfitting khi stacking nhiều layers
- Dropout **trước fc**: Regularization cho classification head
- **Không** dropout sau layer cuối cùng (trước pooling) để giữ thông tin

---

## 10. Tổng kết

| Thành phần | Vai trò |
|-----------|---------|
| **OwnLSTMCell** | Đơn vị cơ bản, xử lý 1 time step với 3 gates |
| **OwnLSTM** | Stack nhiều LSTM cells, unidirectional, average pooling |
| **SelfAttention** | Học trọng số cho từng time step, thay thế average pooling |
| **AttentionLSTM** | LSTM + Attention, tập trung vào từ quan trọng |
| **OwnBiLSTM** | LSTM 2 hướng, context đầy đủ hơn |
| **AttentionBiLSTM** | BiLSTM + Attention, mô hình mạnh nhất |

**Luồng dữ liệu chung:**
```
Text → Tokens → Embedding → LSTM/BiLSTM → Pooling (Avg/Attention) → FC → Prediction
```
