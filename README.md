# 目標
透過自己實現之Linear Discriminant Analysis (LDA)演算法對鳶尾花資料(Iris data set)進行分類；並利用ROC曲線和AUC觀察調整懲罰權重對模型表現所造成的影響；最後，進一步測試多類別分類策略（e.g. One against one strategy）。

# 資料描述
1. 安德森鳶尾花卉數據集(Anderson's Iris data set)為機器學習領域中，常被用來驗證演算法效能的資料庫。數據庫中包含三種不同鳶尾花標籤(Label)：山鳶尾(Setosa)、變色鳶尾(Versicolor)以及維吉尼亞鳶尾(Virginica)，且每種鳶尾花皆有50筆樣本。而每筆樣本以四種屬性作為特徵(單位：cm)：花萼長度(Sepal length)、花萼寬度(Sepal width)、花瓣長度(Petal length)以及花瓣寬度(Petal width)等四種屬性作為定量分析之數據。
2. 讀取鳶尾花資料後會產生150×5的陣列，其中第5行為資料的類別標籤。

# 作業內容
1. 利用程式實現LDA分類器（勿直接使用開源的LDA套件）
2. 利用2-fold cross validation推估LDA在二元分類之分類率
3. 繪製ROC與計算AUC
4. 多類別分類問題 - One against one strategy
5. 討論結果

# 討論
1. ROC與AUC：從confusion matrix可以得知，分類率不同會有不同的TPR以及FPR。當c1/c2比值低時，TPR與FPR會下降，但下降的幅度不一樣，表示c1/c2比值會不但會影響分類率，還會因為影響到bias而decision function較靠近某個類別，發生過擬合。
2. 多類別分類：多類別分類時有些class組合的bias會比較高，雖然較高的bias表示模型的性能較差，其分類率會較低。但因為test data中包含三種類別，所以在此個part所記錄的bias比較沒有可參考性。

# 心得
透過LDA分類器做二元分類，可以發現class 1跟class 3呈現線性可分離，分類率是呈現100%，測試出來的時候還以為是自己程式設計錯誤了。而這次花最多時間的部分是在繪製ROC與計算AUC，原本想要用sklearn的套件一次解決，但遇到的困難是沒辦法在直接不斷更動c1/c2值後的ROC圖形合併成一個曲線。所以最後索性自己花點時間將各個c1/c2值的TPR和FPR做計算，接著再運用結果來繪製ROC。
