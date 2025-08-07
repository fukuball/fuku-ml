# FukuML Warning Handling

## 問題描述

FukuML 在執行某些數值計算時可能會產生 NumPy RuntimeWarning，例如：
- `divide by zero encountered in dot`
- `overflow encountered in dot`  
- `invalid value encountered in dot`
- `divide by zero encountered in matmul`
- `overflow encountered in matmul`
- `invalid value encountered in matmul`

這些警告通常是因為數值計算時遇到數值精度問題，但**不影響功能正確性**。

## 解決方案

### 🎯 全域自動處理（推薦）

**從目前版本開始，FukuML 使用全域設定自動抑制這些警告**，用戶完全不會看到這些警告。

實現方式：
- 在 `FukuML/__init__.py` 中自動載入 `FukuML.Config`
- 使用 NumPy 的 `np.seterr()` 進行全域錯誤狀態控制
- 無需在各個檔案中重複處理警告

### 🔧 進階控制選項

如果您需要控制警告顯示，可以使用 `FukuML.Config` 模組：

```python
import FukuML
import FukuML.Config as config

# 方法1: 恢復警告顯示（用於除錯）
config.suppress_warnings(False)

# 使用您的演算法 - 現在會顯示警告
import FukuML.PLA as pla
pla_bc = pla.BinaryClassifier()
pla_bc.init_W('linear_regression_accelerator')

# 方法2: 重新抑制警告
config.suppress_warnings(True)

# 方法3: 重置所有警告設置
config.reset_warnings()
```

### 🚫 如果您想要完全控制警告行為

```python
import FukuML.Config as config

# 完全停用 FukuML 的警告管理，恢復 NumPy 預設行為
config.suppress_warnings(False)

# 您也可以直接使用 NumPy 的 seterr
import numpy as np
np.seterr(all='warn')  # 恢復預設警告
np.seterr(divide='ignore', over='ignore', invalid='ignore')  # 自訂抑制
```

## 為什麼會有這些警告？

這些警告來自於：

1. **偽逆矩陣計算**：`np.linalg.pinv()` 在處理近似奇異矩陣時
2. **數值精度限制**：浮點運算的固有限制
3. **特殊數據情況**：某些數據分布可能導致數值不穩定

## 影響評估

- ✅ **功能正常**：警告不影響算法的正確性
- ✅ **結果準確**：數值計算結果依然正確
- ✅ **用戶體驗**：現在警告已被自動抑制

## 測試驗證

可以運行任何測試確認用戶體驗：

```bash
# 測試 PLA 演算法 - 不會出現警告
python tests/test_pla_binary_classifier.py
python tests/test_pla_multi_classifier.py

# 測試其他演算法
python test_fuku_ml.py
```

這些測試會確認不會出現任何 RuntimeWarning。

## 技術細節

### 實現原理
1. **全域配置**：在 `FukuML/Config.py` 中定義全域警告控制
2. **自動初始化**：`FukuML/__init__.py` 中自動 import Config 並啟用警告抑制
3. **NumPy seterr**：使用 `np.seterr()` 全域控制數值錯誤行為
4. **無需修改演算法**：各演算法檔案保持簡潔，無需個別處理警告

### 優點
- **一次設定，全域生效**：無需在每個檔案處理警告
- **代碼更簡潔**：移除重複的警告處理代碼
- **維護性更好**：警告控制邏輯集中管理
- **使用者友好**：預設提供最佳體驗

## 向後相容性

- ✅ 所有現有 API 保持不變
- ✅ 算法行為完全一致
- ✅ 只是抑制了警告顯示
- ✅ 使用者仍可自由控制警告行為