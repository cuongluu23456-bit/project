# Tối ưu nhập hàng cho cửa hàng bán lẻ: Dự báo nhu cầu (ML) + Tối ưu (thuật toán AI cổ điển)

> **Luận điểm của dự án:** quyết định nhập hàng tốt cần **hai thứ cùng lúc**: một dự báo nhu cầu tốt
> (Machine Learning) và một bộ tối ưu biết dùng dự báo đó trong điều kiện **bất định** và **ngân sách/kho có hạn**
> (Genetic Algorithm, Quy hoạch động, Tham lam). Dự án đo xem mỗi mảnh đóng góp bao nhiêu so với cách nhập thông thường.

Ngôn ngữ: Python. Thư viện: chỉ `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
**Greedy, Quy hoạch động và Thuật toán di truyền đều tự viết từ đầu** (không dùng `scipy.optimize`, `pulp`, `deap`...).
Toàn bộ tên hàm, biến, chú thích bằng tiếng Việt để dễ đọc và học.

---

## 1. Bài toán

Mỗi ngày cửa hàng phải quyết định **nhập bao nhiêu** của mỗi mặt hàng (20 mặt hàng) cho ngày hôm sau, khi:

* nhu cầu ngày mai **chưa biết** (chỉ dự báo được, và dự báo luôn có sai số);
* **ngân sách** nhập hàng có hạn (B) và **kho** chứa có hạn (V);
* nhập thừa thì tốn phí tồn kho, nhập thiếu thì mất doanh thu;
* nhập mỗi mặt hàng có **phí cố định** (vận chuyển, đặt hàng...) và chỉ được nhập **theo thùng**.

### Quy trình ba bước

```
Dữ liệu bán hàng ──► [1] ML dự báo nhu cầu ──► [2] Sinh 200 kịch bản nhu cầu ──► [3] Tối ưu số thùng nhập
 (theo ngày)         Random Forest /            (dự báo + sai số bootstrap)       Greedy / DP / Thuật toán di truyền
                     Gradient Boosting                                            ──► kế hoạch nhập + cảnh báo thiếu hàng
```

Đây là cách làm "**dự báo rồi tối ưu**" (predict-then-optimize): ML trả lời *"nhu cầu khoảng bao nhiêu?"*,
thuật toán tối ưu trả lời *"vậy nên nhập bao nhiêu để tổng chi phí kỳ vọng thấp nhất?"*.

### Công thức chi phí (hàm mục tiêu)

Quyết định: `so_thung_i` (số nguyên ≥ 0); số lượng nhập `q_i = so_thung_i × quy_cach_i`.

```
Q(q) =  Σ_i [ c_i·q_i + K_i·1(q_i>0) ]                       ← chi phí nhập hàng (+ phí cố định mỗi lần nhập)
      + (1/S) Σ_s Σ_i [ h_i·max(q_i − d_is, 0)               ← chi phí tồn kho: mỗi đơn vị DƯ
                      + P_i·max(d_is − q_i, 0) ]             ← chi phí thiếu hàng: doanh thu bị MẤT

với ràng buộc:   Σ_i c_i·q_i ≤ B        (ngân sách)        Σ_i v_i·q_i ≤ V        (sức chứa kho)
```

| Ký hiệu | Tên trong code | Ý nghĩa |
|---|---|---|
| `P_i` | `gia_ban` | giá bán (lấy từ dữ liệu) |
| `c_i` | `gia_nhap` | giá nhập (**giả định**: 35–55% giá bán) |
| `h_i` | `phi_ton_kho` | phí tồn kho mỗi đơn vị dư (**giả định**: 3% giá nhập) |
| `K_i` | `phi_co_dinh` | phí cố định mỗi lần nhập (**giả định**: 4 × giá nhập) |
| `v_i` | `the_tich` | thể tích kho mỗi đơn vị (**giả định**: 0.5–3.0) |
| `k_i` | `quy_cach` | số đơn vị trong một thùng (**giả định**: 1, 6 hoặc 12 tùy mặt hàng bán nhiều hay ít) |
| `d_is` | `kich_ban` | nhu cầu của mặt hàng i trong kịch bản s (S = 200) |
| `B`, `V` | `ngan_sach`, `suc_chua` | đặt bằng 80% mức cần để nhập đủ nhu cầu trung bình |

**Vì sao thiếu hàng tính bằng cả giá bán `P_i` chứ không phải lãi `P_i − c_i`?** Vì trong công thức, tiền nhập `c_i·q_i` đã được tính riêng
ở số hạng đầu. Lợi nhuận = `P·d − [chi phí nhập + tồn kho + P·(thiếu)]`, nên cực tiểu hóa Q chính là cực tiểu hóa chi phí
cơ hội và đúng bằng cực đại hóa lợi nhuận (`P·d` là hằng số).

**Kiểm tra đúng sai (có trong test):** nếu không có ràng buộc, không phí cố định, quy cách = 1, thì nghiệm tối ưu của từng mặt hàng
là **phân vị mức `(P − c)/(P + h)`** của các kịch bản nhu cầu (công thức *người bán báo* / newsvendor kinh điển).
Phí cố định, quy cách thùng và hai ràng buộc chung là những yếu tố làm bài toán *không còn có công thức đóng*, đó là chỗ thuật toán tối ưu có việc để làm.

> **Mẹo cài đặt:** Q cộng **tách** theo từng mặt hàng, nên `bai_toan.py` tính trước bảng `bang_chi_phi[i][g]`
> (chi phí kỳ vọng của mặt hàng i nếu nhập g thùng). Sau đó cả ba thuật toán chỉ việc *tra bảng*, rất nhanh.

---

## 2. Cấu trúc thư mục

```
nhap_hang_ai/
├── README.md                     ← file này
├── requirements.txt
├── cau_hinh.py                   ← MỌI tham số giả định nằm ở đây (đọc file này đầu tiên)
├── du_lieu.py                    ← đọc dữ liệu; sinh dữ liệu GIẢ nếu chưa có dữ liệu thật
├── chuan_bi_du_lieu_uci.py       ← biến file UCI Online Retail thành CSV chuẩn (chạy 1 lần, 1 người)
├── du_bao.py                     ← PHẦN 1 (ML): đặc trưng, chia dữ liệu, mô hình, kịch bản nhu cầu
├── bai_toan.py                   ← PHẦN 2: định nghĩa bài toán, hàm chi phí, ràng buộc, bảng chi phí
├── tham_lam.py                   ← PHẦN 2: thuật toán Tham lam (baseline)
├── quy_hoach_dong.py             ← PHẦN 2: Quy hoạch động (nghiệm chính xác trên lưới rời rạc)
├── di_truyen.py                  ← PHẦN 2: Thuật toán di truyền (điểm nhấn)
├── thuc_nghiem.py                ← các thực nghiệm E1, E2, E3 và kế hoạch ngày mai
├── ve_hinh.py                    ← vẽ 5 biểu đồ
├── chay_demo.py                  ← chạy TOÀN BỘ dự án bằng một lệnh
├── tests/kiem_thu.py             ← 18 kiểm thử tự động
├── data/                         ← đặt file dữ liệu thật vào đây
└── ket_qua/                      ← kết quả sinh ra (bảng CSV + biểu đồ PNG)
```

---

## 3. Cài đặt

Cần Python 3.9 trở lên (đã chạy thử với Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, matplotlib 3.11).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Dữ liệu

Dự án cần một file `data/ban_hang_theo_ngay.csv` với **5 cột**: `ngay, ma_hang, ten_hang, so_luong, don_gia`
(mỗi dòng = một mặt hàng trong một ngày mở cửa; `so_luong` = 0 nếu hôm đó không bán được).

### Cách 1: chạy thử ngay, không cần tải gì
Nếu **không có** file trên, chương trình tự sinh **dữ liệu GIẢ** (20 mặt hàng, 283 ngày, có quy luật theo thứ trong tuần,
xu hướng và khuyến mãi) và **in cảnh báo rất rõ** ở đầu và cuối.

> ⚠️ **Dữ liệu giả chỉ để kiểm tra chương trình chạy đúng. Số liệu từ dữ liệu giả KHÔNG được đưa vào báo cáo như kết quả thật.**

### Cách 2: dùng dữ liệu thật (làm trước khi chốt báo cáo)
Bộ dữ liệu **UCI Online Retail** (giao dịch của một cửa hàng online ở Anh, 12/2010 – 12/2011):
<https://archive.ics.uci.edu/dataset/352/online-retail>

```bash
pip install openpyxl                                   # chỉ cần để đọc file Excel .xlsx gốc
python chuan_bi_du_lieu_uci.py "Online Retail.xlsx"    # tạo data/ban_hang_theo_ngay.csv
```

Script sẽ: bỏ hóa đơn hủy / số lượng ≤ 0 / giá ≤ 0, chỉ giữ mã sản phẩm thật, gộp theo ngày, chọn 20 mặt hàng bán đều nhất,
điền 0 cho ngày mở cửa mà mặt hàng không bán, và cắt ngoại lệ (đơn sỉ rất lớn) ở phân vị 99.
Chỉ **một người** cần chạy bước này; file CSV đầu ra nhỏ, nên commit lên git cho cả nhóm cùng dùng.

> **Lưu ý quan trọng về dữ liệu UCI:** nó chỉ có số lượng và giá bán. **Không có** giá vốn, thể tích kho, phí tồn kho, phí cố định, quy cách thùng,
> nên các tham số đó là **giả định của nhóm** (xem `cau_hinh.py`). Báo cáo phải nói rõ điều này. Ngoài ra nhiều khách mua là nhà bán sỉ,
> nên nhu cầu có những đỉnh lớn; đó là lý do script cắt ở phân vị 99.
> Script đã được kiểm thử trên một file giả đúng định dạng UCI (mã hóa latin-1, có hóa đơn hủy, mã POST...).
> Nhóm nên chạy nó trên file thật và xem lại bảng đầu ra.

---

## 5. Chạy

```bash
python chay_demo.py                 # chạy đủ, khoảng 40 giây
python chay_demo.py --nhanh         # bản rút gọn để thử, khoảng 10 giây
python chay_demo.py --du-lieu duong/dan/den/file.csv
python tests/kiem_thu.py            # chạy 18 kiểm thử (khoảng 10 giây)
```

`chay_demo.py` in ra màn hình và lưu vào `ket_qua/`:

| File | Nội dung |
|---|---|
| `bang_E1_du_bao.csv`, `hinh_1_du_bao.png` | E1: sai số dự báo của các mô hình; dự báo so với thực tế |
| `ke_hoach_ngay_mai.csv`, `ke_hoach_ngay_mai_so_sanh.csv`, `hinh_2_ke_hoach_ngay_mai.png` | **kế hoạch nhập hàng ngày mai** theo từng phương pháp, xác suất thiếu, cảnh báo |
| `bang_E2_toi_uu_mot_ngay.csv`, `hinh_3_E2_chi_phi.png`, `hinh_4_hoi_tu_ga.png` | E2: so sánh thuật toán ở nhiều mức ràng buộc; đường hội tụ của GA |
| `bang_E3_backtest_*.csv`, `hinh_5_backtest.png` | E3: chạy lại 40 ngày trong quá khứ với nhu cầu thật |

Bảng in ra màn hình cùng các biểu đồ này thay cho "dashboard" (nhóm đã quyết định bỏ giao diện web): cùng thông tin nhưng chỉ cần `matplotlib`.

---

## 6. Từng module làm gì

### `du_bao.py`: Phần ML
* **Đặc trưng** (chỉ dùng thông tin biết *trước* ngày cần dự báo): mặt hàng, thứ trong tuần, giá bán, doanh số lần mở cửa trước (`lag_1`),
  doanh số cùng thứ tuần trước (`lag_7`), trung bình 7 lần mở cửa gần nhất (`tb_7`). **Tuyệt đối không dùng doanh số của chính ngày cần dự báo** (rò rỉ dữ liệu); có test kiểm tra điều này.
* **Chia theo thời gian**, không chia ngẫu nhiên: 60% ngày đầu để huấn luyện, 20% tiếp theo để chọn mô hình và đo sai số (valid), 20% cuối để kiểm tra (test).
* **So sánh** hai mô hình ML (Random Forest, Gradient Boosting) với hai cơ sở đơn giản ("bằng tuần trước", "trung bình 7 ngày").
  Mô hình ML phải thắng cơ sở thì mới đáng dùng. Chọn mô hình tốt nhất theo MAE trên tập valid.
* **Kịch bản nhu cầu**: dự báo chỉ là *một con số*, còn nhu cầu thật dao động quanh nó. Ta lấy **sai số dự báo trên tập valid** của từng mặt hàng,
  bốc mẫu lại có hoàn lại (*residual bootstrap*) rồi cộng vào dự báo, được 200 kịch bản nhu cầu. Nhờ vậy chi phí tồn kho/thiếu hàng mới có ý nghĩa.

### `bai_toan.py`: Định nghĩa bài toán
Lớp `BaiToan` giữ giá, ràng buộc, kịch bản; cung cấp `chi_phi`, `hop_le`, `xac_suat_thieu`, hai cách sửa nghiệm vi phạm ràng buộc
(`sua_nghiem_co_ty_le`, `sua_nghiem_tham_lam`) và `chi_phi_thuc_te` (đánh giá khi đã biết nhu cầu thật, dùng cho backtest).

### `tham_lam.py`: Greedy (baseline)
Bắt đầu từ "không nhập gì", mỗi bước thêm **một thùng** cho mặt hàng có *lợi ích giảm chi phí trên mỗi đơn vị nguồn lực* lớn nhất, dừng khi hết ngân sách/kho hoặc không còn thùng nào có lợi.
Có 3 cách chấm điểm nguồn lực (`tong_chuan_hoa`, `ngan_sach`, `tuyet_doi`); thực nghiệm báo cáo cả ba.
**Điểm yếu:** chỉ nhìn một bước. Với phí cố định `K_i`, thùng đầu tiên của một mặt hàng có thể "chưa có lợi" (còn gánh `K_i`) dù các thùng sau rất có lợi, nên Greedy bỏ lỡ mặt hàng đó. Điều này hiện rõ ở E2 khi ràng buộc lỏng.

### `quy_hoach_dong.py`: Quy hoạch động
Bài toán balo nhiều chiều (hai ràng buộc, mỗi mặt hàng chọn 0…G thùng). Trạng thái `f[b][v]` = chi phí nhỏ nhất khi đã xét các mặt hàng đầu tiên và dùng tối đa `b` ô ngân sách, `v` ô kho.
Ngân sách và kho được chia thành lưới 500 × 300 ô. **Nghiệm luôn hợp lệ** (làm tròn lên số ô mỗi mặt hàng chiếm) nhưng chỉ **tối ưu trên lưới**, có thể "lãng phí" tối đa 1 ô mỗi mặt hàng. Độ phức tạp `~ n × ô_tiền × ô_kho × số thùng`: thêm một ràng buộc nữa là thêm một chiều của bảng, rất nặng.

### `di_truyen.py`: Thuật toán di truyền (điểm nhấn)
* **Nhiễm sắc thể:** vector số thùng của 20 mặt hàng. **Độ thích nghi:** chi phí kỳ vọng Q (càng nhỏ càng tốt).
* **Chọn lọc giải đấu** (bốc 3 cá thể, lấy người tốt nhất) → **lai ghép đồng nhất** (xác suất 0.9) → **đột biến** (tăng/giảm 1–2 thùng, hoặc đặt lại ngẫu nhiên) → **giữ lại 2 cá thể ưu tú** nên kết quả không bao giờ xấu đi.
* **Xử lý ràng buộc** hai cách để so sánh: `"sua"` (mặc định: cá thể vượt ngân sách/kho bị co đồng loạt về mức hợp lệ) và `"phat"` (cộng một khoản phạt lớn). Cuối cùng nghiệm tốt nhất luôn được sửa nếu còn vi phạm.
* Quần thể 150 cá thể × 400 thế hệ (khoảng 0.07 giây mỗi lần chạy). **GA ngẫu nhiên**: mỗi `hat` (seed) cho kết quả hơi khác, nên thực nghiệm chạy **30 seed** và báo cáo **trung bình ± độ lệch chuẩn**.

### `thuc_nghiem.py`: Ba thực nghiệm
| | Câu hỏi | Cách làm |
|---|---|---|
| **E1** | ML có thắng cơ sở đơn giản không? | MAE và WAPE (sai số tuyệt đối tổng / tổng doanh số) trên valid và test |
| **E2** | Thuật toán nào tối ưu tốt nhất, và khi nào? | Một ngày (ngày đầu tập test), ràng buộc ở 30%, 50%, 80%, 150% mức cần. Báo cáo chi phí kỳ vọng, % cao hơn DP, thời gian. GA 30 seed |
| **E3** | Dùng cả hệ thống có tiết kiệm tiền *thật* không? | Backtest 40 ngày của tập test: lập kế hoạch bằng dự báo, rồi áp **nhu cầu thật** để tính chi phí thực tế, tỉ lệ đáp ứng, số mặt hàng thiếu |

Các phương pháp được so sánh: **Nhập thông thường** (nhập bằng doanh số cùng thứ tuần trước, nếu vượt ngân sách/kho thì cắt đồng loạt theo tỉ lệ), **Tham lam**, **Di truyền**, **Quy hoạch động**.
Riêng E3 có thêm **Dự báo hoàn hảo** (biết trước nhu cầu thật, giải bằng DP): không phải phương pháp dùng được, mà là *mốc tham chiếu* cho thấy dự báo tốt hơn có thể tiết kiệm tối đa bao nhiêu.

### Kế hoạch ngày mai (sản phẩm cho người ra quyết định)
Dự báo cho ngày mở cửa kế tiếp → sinh kịch bản → giải bằng cả bốn phương pháp → **đề xuất phương án có chi phí kỳ vọng thấp nhất trong các phương án hợp lệ**.
Với mỗi mặt hàng có **xác suất thiếu hàng** = tỉ lệ kịch bản có nhu cầu lớn hơn số lượng nhập; cảnh báo nếu ≥ 80% (`NGUONG_NGUY_CO_THIEU` trong `cau_hinh.py`).

---

## 7. Ví dụ kết quả (trên DỮ LIỆU GIẢ, chỉ để biết kết quả trông như thế nào)

> ⚠️ Các con số dưới đây đến từ dữ liệu giả với `HAT_NGAU_NHIEN = 42`. **Phải chạy lại trên dữ liệu thật** và dùng số đó trong báo cáo.
> Điều đáng quan tâm là *hình dạng* của kết quả (cái nào hơn cái nào, hơn bao nhiêu), không phải con số cụ thể.

**E1: dự báo (tập test)**

| Phương pháp | MAE | WAPE |
|---|---|---|
| Bằng tuần trước (cơ sở) | 11.34 | 52.2% |
| Trung bình 7 ngày (cơ sở) | 9.90 | 45.6% |
| Random Forest | **8.74** | **40.2%** |
| Gradient Boosting | 8.99 | 41.4% |

**E2: chi phí kỳ vọng cao hơn DP bao nhiêu % (thấp hơn = tốt hơn; GA là trung bình 30 seed)**

| Ngân sách & kho (so với mức cần) | Nhập thông thường | Tham lam (tốt nhất trong 3 cách) | Di truyền (GA) |
|---|---|---|---|
| 30% | +12.5% | +1.1% | −0.04% |
| 50% | +14.9% | +0.5% | +0.04% |
| 80% | +18.8% | +2.5% | −0.11% |
| 150% | +20.4% | +5.4% | 0.00% |

**E3: backtest 40 ngày (chi phí thực tế khi đã biết nhu cầu thật; ngân sách và kho = 80% mức cần)**

| Phương pháp | Chi phí TB/ngày | Tỉ lệ đáp ứng | Tiết kiệm so với thông thường |
|---|---|---|---|
| Nhập thông thường | 1127.7 | 54.2% | 0% |
| Tham lam | 1074.0 | 61.1% | 4.8% |
| Di truyền (GA) | 1048.0 | 67.4% | 7.1% |
| Quy hoạch động (DP) | 1050.1 | 66.6% | 6.9% |
| *Dự báo hoàn hảo (mốc tham chiếu)* | *871.8* | *77.9%* | *22.7%* |

### Cách đọc kết quả (đừng nói quá những gì dữ liệu cho thấy)

1. **GA ≈ DP.** Chênh lệch trong E2 chỉ cỡ ±0.1%, nhỏ hơn cả độ lệch chuẩn giữa các seed ở một số ô. Có lúc GA *nhỉnh hơn* DP một chút (số âm) vì DP chỉ tối ưu trên lưới và làm tròn lên. Vì vậy **không kết luận "GA tốt hơn DP"**; chỉ nói được "GA đạt chất lượng ngang DP". Tương tự, ở E3 GA và DP chênh nhau ~0.2%, nằm trong nhiễu của 40 ngày.
2. **Cả hai đều hơn Greedy rõ rệt, và khoảng cách phụ thuộc vào độ chặt của ràng buộc.** Khi ràng buộc chặt (30–50%) Greedy chỉ kém 0.5–1.1%, vì chuyện chính là *chọn mặt hàng nào*, mà "lợi ích trên nguồn lực" làm việc đó khá tốt. Khi ràng buộc lỏng (150%) Greedy kém 5.4%: lúc này ngân sách không còn là vấn đề, cái khó là **phí cố định** và **quy cách thùng** làm Greedy (chỉ nhìn một bước) bỏ lỡ các mặt hàng đáng nhập.
3. **Nhập thông thường kém hơn 12–20%** về chi phí kỳ vọng, vì nó không tính đến sự bất định (không biết nên nhập thừa hay thiếu bao nhiêu) và cắt đồng loạt theo tỉ lệ thay vì chọn mặt hàng.
4. **Khoảng cách tới "dự báo hoàn hảo" lớn hơn nhiều so với khoảng cách giữa các thuật toán** (E3: hệ thống tiết kiệm ~7%, dự báo hoàn hảo tiết kiệm ~23%). Nghĩa là ở bài toán này, **chất lượng dự báo (ML) là nút thắt chính**, thuật toán tối ưu đã gần chạm trần. Đây là một kết luận hay để đưa vào phần thảo luận.
5. **Phần tiết kiệm 7% ở E3 gộp hai hiệu ứng:** dự báo ML tốt hơn "tuần trước" *và* tối ưu có tính bất định. Dự án chưa tách riêng hai hiệu ứng này (xem mục *Hướng mở rộng*).
6. Tỉ lệ đáp ứng chỉ 54–67% vì ngân sách/kho chỉ bằng 80% mức cần và dữ liệu giả có nhiễu mạnh (WAPE ~40%). Con số trên dữ liệu thật sẽ khác.

---

## 8. Hạn chế (nên viết thẳng vào báo cáo)

* **Tham số chi phí là giả định** (giá vốn, thể tích, phí tồn kho, phí cố định, quy cách). Kết luận về *thuật toán nào hơn* khá bền với các giả định này; kết luận về *số tiền tiết kiệm* thì không.
  Có thể thử nhiều giá trị trong `cau_hinh.py` để xem kết luận có đổi không (phân tích độ nhạy).
* **Mô hình một chu kỳ**: mỗi ngày bắt đầu với tồn kho 0, giao hàng ngay (lead time = 0), hàng dư không mang sang ngày sau mà chỉ chịu phí tồn kho. Thực tế phức tạp hơn nhiều (tồn kho đầu kỳ, thời gian giao hàng, hạn sử dụng...).
* **Giả định về sai số dự báo**: kịch bản được tạo bằng cách bốc lại sai số quá khứ của *từng mặt hàng*, tức coi sai số độc lập giữa các ngày và có phân bố giống nhau. Nếu sai số tăng vào ngày khuyến mãi hay dịp lễ thì kịch bản sẽ đánh giá thấp rủi ro.
* **Dữ liệu UCI**: là cửa hàng online, nhiều khách sỉ, nhu cầu nhiều đỉnh nhọn; chỉ một năm dữ liệu nên chưa có mùa vụ năm thứ hai; một số ngày lễ cửa hàng đóng cửa cả tuần.
* **Backtest 40 ngày** là một mẫu nhỏ. Nên báo cáo kèm cột "% ngày rẻ hơn nhập thông thường" (đã có trong bảng E3 đầy đủ) và không khẳng định chênh lệch dưới 1%.
* **DP rời rạc hóa** nên không phải nghiệm tối ưu tuyệt đối; lưới càng mịn càng gần nhưng càng chậm.

### Hướng mở rộng (nếu còn thời gian)
* Tách đóng góp: chạy GA nhưng thay dự báo ML bằng dự báo "bằng tuần trước" để biết riêng phần ML đóng góp bao nhiêu.
* Phân tích độ nhạy theo `TI_LE_GIA_VON`, `HE_SO_PHI_CO_DINH`, `TI_LE_PHI_TON_KHO`.
* Thêm ràng buộc thứ ba (ví dụ số mặt hàng tối đa được nhập mỗi ngày). DP sẽ rất nặng, GA thì sửa vài dòng.

---

## 9. Câu hỏi hội đồng có thể hỏi

**Nếu DP cho nghiệm chính xác rồi, sao còn cần GA?**
DP chỉ chính xác *trên lưới rời rạc*, và mỗi ràng buộc thêm vào là thêm một chiều của bảng (bộ nhớ và thời gian nhân lên). GA không bị chuyện đó: thêm ràng buộc hay đổi hàm chi phí đều chỉ cần sửa hàm chấm điểm. E2 cho thấy GA đạt chất lượng ngang DP, và thêm rằng GA ngẫu nhiên nên phải báo cáo nhiều seed.

**Sao không dùng một bộ giải tối ưu có sẵn (scipy, pulp)?**
Mục tiêu môn học là hiểu thuật toán, nên ba thuật toán đều tự cài. Ta dùng DP làm *mốc đối chiếu* cho GA, và dùng vét cạn trong test để kiểm tra DP đúng.

**Greedy có luôn tệ hơn không?**
Không. Ở ràng buộc chặt, Greedy chỉ kém DP khoảng 0.5–1.1%. Ưu điểm của nó là đơn giản và gần như tức thì. Nó chỉ thua rõ khi phí cố định và quy cách thùng làm bài toán "không lồi".

**Làm sao biết dự báo không "nhìn trộm" tương lai?**
Đặc trưng chỉ lấy từ quá khứ (có test đổi doanh số ngày t thành số rất lớn thì đặc trưng của ngày t không đổi), và dữ liệu chia theo thời gian chứ không chia ngẫu nhiên.

**Vì sao kết quả của em khác kết quả trong README?**
README dùng dữ liệu giả. Dữ liệu thật cho số khác; điều cần kiểm tra lại là các *xu hướng* (GA ≈ DP > Greedy > thông thường).

---

## 10. Thuật ngữ

| Thuật ngữ | Giải thích ngắn |
|---|---|
| MAE | sai số tuyệt đối trung bình: trung bình của \|thực tế − dự báo\| |
| WAPE | tổng sai số tuyệt đối chia tổng doanh số; dễ hiểu hơn MAE khi so sánh mặt hàng bán nhiều/ít |
| Rò rỉ dữ liệu | mô hình vô tình dùng thông tin mà thực tế chưa biết tại thời điểm dự báo, làm kết quả đẹp giả tạo |
| Kịch bản | một "tương lai có thể xảy ra" của nhu cầu; có 200 kịch bản mỗi ngày |
| Residual bootstrap | bốc ngẫu nhiên có hoàn lại các sai số quá khứ để mô phỏng sự bất định |
| Newsvendor | bài toán "người bán báo": nhập bao nhiêu khi nhu cầu bất định và hàng không bán được là thành phí |
| Knapsack (balo) | chọn tập vật phẩm giá trị lớn nhất mà tổng "trọng lượng" không vượt sức chứa |
| Phí cố định | khoản phí trả mỗi lần nhập một mặt hàng, bất kể nhập nhiều hay ít |
| Elitism (ưu tú) | giữ nguyên vài cá thể tốt nhất sang thế hệ sau |
| Backtest | chạy lại hệ thống trên quá khứ (với nhu cầu đã biết) để đo hiệu quả thực |
| Seed (`hat`) | số khởi tạo bộ sinh ngẫu nhiên; cùng seed cho cùng kết quả |

---

## 11. Gợi ý phân công và lịch 6 ngày

| Việc | Module | Ghi chú |
|---|---|---|
| Dữ liệu thật (UCI), kiểm tra bảng đầu ra, viết phần "Dữ liệu" của báo cáo | `chuan_bi_du_lieu_uci.py`, `du_lieu.py` | làm **ngay ngày 1**, vì mọi số liệu sau đó phụ thuộc vào đây |
| Phần ML: hiểu đặc trưng, đọc kết quả E1, viết phần dự báo | `du_bao.py` | |
| Bài toán + Greedy + DP, viết phần mô hình hóa | `bai_toan.py`, `tham_lam.py`, `quy_hoach_dong.py` | |
| GA + E2, viết phần thuật toán di truyền | `di_truyen.py`, `thuc_nghiem.py` | |
| Backtest E3, biểu đồ, thảo luận, slide | `thuc_nghiem.py`, `ve_hinh.py` | |

* **Ngày 1:** cài đặt, chạy `chay_demo.py` trên dữ liệu giả để cả nhóm thấy dự án chạy; một người chuẩn bị dữ liệu UCI.
* **Ngày 2:** chạy lại toàn bộ trên dữ liệu thật; mỗi người đọc phần code của mình, chạy `python tests/kiem_thu.py`.
* **Ngày 3:** thử đổi tham số trong `cau_hinh.py` (độ nhạy); chốt các giả định.
* **Ngày 4:** chốt số liệu và biểu đồ cuối; bắt đầu viết báo cáo.
* **Ngày 5:** hoàn thiện báo cáo và slide; tập trả lời các câu hỏi ở mục 9.
* **Ngày 6:** dự phòng, rà soát lần cuối.

Mỗi thành viên nên **tự giải thích được phần của mình** (và đọc lướt các phần còn lại): hội đồng thường hỏi theo từng người.

---

## 12. Tài liệu tham khảo

* Arrow, K. J., Harris, T., Marschak, J. (1951). *Optimal Inventory Policy*. Econometrica 19(3). (gốc của bài toán newsvendor)
* Nahmias, S. *Production and Operations Analysis*. McGraw-Hill. (chương về mô hình tồn kho một chu kỳ)
* Elmachtoub, A. N., Grigas, P. (2022). *Smart "Predict, then Optimize"*. Management Science 68(1). (cách làm dự báo rồi tối ưu)
* Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
* Eiben, A. E., Smith, J. E. *Introduction to Evolutionary Computing*. Springer. (chọn lọc giải đấu, xử lý ràng buộc trong GA)
* Kellerer, H., Pferschy, U., Pisinger, D. (2004). *Knapsack Problems*. Springer. (quy hoạch động cho bài toán balo)
* Breiman, L. (2001). *Random Forests*. Machine Learning 45. · Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics 29(5).
* Chen, D., Sain, S. L., Guo, K. (2012). *Data mining for the online retail industry*. Journal of Database Marketing & Customer Strategy Management 19. · Bộ dữ liệu: UCI Online Retail, <https://archive.ics.uci.edu/dataset/352/online-retail>.
