________________________________________
	Data Loading and Importing
________________________________________
 		CSV Files:
			read.csv(), readr::read_csv(), utils::read.table()
			data.table::fread()
		Excel Files:
			readxl::read_excel(), openxlsx::read.xlsx()
			writexl::write_xlsx()
		SQL Databases:
			DBI::dbConnect(), RSQLite::dbConnect(), RMySQL::dbConnect()
			dplyr::tbl()
			sqldf::sqldf()
		JSON Files:
			jsonlite::fromJSON()
			rjson::fromJSON()
			HDF5 Files: rhdf5::h5read()
		Parquet Files:
			arrow::read_parquet()
		Big Data:
			bigmemory::read.big.matrix(), bigstatsr::fread()
		Web Scraping:
			rvest::read_html()
			httr::GET()
			RCurl::getURL()
________________________________________
	Data Cleaning and Preprocessing
________________________________________
		Handling Missing Data:
			Identify Missing Data: is.na(), summary(), anyNA()
			Removing Missing Data: na.omit(), complete.cases(), drop_na() (from tidyr)
		Imputation:
			Mean Imputation: dplyr::mutate() with mean()
			Median Imputation: dplyr::mutate() with median()
			Mode Imputation: modeest::mfv()
			KNN Imputation: VIM::kNN()
			Multiple Imputation (MICE): mice::mice(), Amelia::amelia()
			Locf Imputation (Last Observation Carried Forward): zoo::na.locf()
________________________________________
		Outlier Detection:
			Z-Score Method: scale()
			IQR Method: IQR(), dplyr::mutate() with IQR()
			Boxplot: boxplot(), ggplot2::geom_boxplot()
			Isolation Forest: isotree::isolation.forest()
			LOF (Local Outlier Factor): DMwR::lof()
			DBSCAN: dbscan::dbscan()
________________________________________
		Duplicated Data:
			duplicated(), unique(), distinct() (from dplyr)
		Feature Engineering:
			One-Hot Encoding: caret::dummyVars(), fastDummies::dummy_cols()
		Label Encoding: forcats::fct_recode()
			Binning: cut(), dplyr::mutate()
			Polynomial Features: poly()
			Interaction Terms: *, : in lm() formula
			Principal Component Analysis (PCA): prcomp(), FactoMineR::PCA()
			t-SNE: Rtsne::Rtsne()
			UMAP: umap::umap()
			Log Transformation: log(), log10()
			Winsorization: DescTools::Winsorize()
________________________________________
	Descriptive Statistics
________________________________________
		Measures of Central Tendency:
			Mean: mean()
			Median: median()
			Mode: modeest::mfv(), table(x), which.max()
			Geometric Mean: DescTools::GeoMean()
			Harmonic Mean: DescTools::HarmMean()
________________________________________
		Measures of Dispersion:
			Variance: var()
			Standard Deviation: sd()
			Range: range()
			Interquartile Range: IQR()
			Mean Absolute Deviation: mad()
			Skewness: e1071::skewness(), moments::skewness()
			Kurtosis: e1071::kurtosis(), moments::kurtosis()
			Coefficient of Variation: cv(), custom function sd()/mean()
________________________________________
		Covariance and Correlation:
			Covariance: cov()
			Pearson Correlation: cor()
			Spearman Rank Correlation: cor(method = "spearman")
			Kendall Tau: cor(method = "Kendall")
			Partial Correlation: ppcor::pcor()
________________________________________
	Probability Distributions
		Discrete Distributions:
			Binomial Distribution:
			PMF: dbinom()
			CDF: pbinom()
			Quantile: qbinom()
			Random Sampling: rbinom()
________________________________________
	Poisson Distribution:
			PMF: dpois()
			CDF: ppois()
			Quantile: qpois()
			Random Sampling: rpois()
________________________________________
	Negative Binomial Distribution:
			PMF: dnbinom()
			CDF: pnbinom()
			Random Sampling: rnbinom()
			Geometric Distribution:
			PMF: dgeom()
			CDF: pgeom()
			Random Sampling: rgeom()
________________________________________
		Multinomial Distribution:
			PMF: dmultinom()
			Random Sampling: rmultinom()
________________________________________
	Continuous Distributions:
________________________________________
		Normal Distribution:
			PDF: dnorm()
			CDF: pnorm()
			Quantile: qnorm()
			Random Sampling: rnorm()
________________________________________
		Uniform Distribution:
			PDF: dunif()
			CDF: punif()
			Random Sampling: runif()
________________________________________
		Exponential Distribution:
			PDF: dexp()
			CDF: pexp()
			Random Sampling: rexp()
________________________________________
		Gamma Distribution:
			PDF: dgamma()
			CDF: pgamma()
			Random Sampling: rgamma()
________________________________________
		Beta Distribution:
			PDF: dbeta()
			CDF: pbeta()
			Random Sampling: rbeta()
________________________________________
		Log-Normal Distribution:
			PDF: dlnorm()
			CDF: plnorm()
		Random Sampling: rlnorm()
________________________________________
		Chi-Square Distribution:
			PDF: dchisq()
			CDF: pchisq()
			Random Sampling: rchisq()
________________________________________
		Student's t-Distribution:
			PDF: dt()
			CDF: pt()
			Random Sampling: rt()
________________________________________
		F-Distribution:
			PDF: df()
			CDF: pf()
			Random Sampling: rf()
________________________________________
		Weibull Distribution:
			PDF: dweibull()
			CDF: pweibull()
			Random Sampling: rweibull()
________________________________________
		Cauchy Distribution:
			PDF: dcauchy()
			CDF: pcauchy()
			Random Sampling: rcauchy()
________________________________________
	Multivariate Distributions:
		Multivariate Normal Distribution:
			PDF: dmvnorm()
			Random Sampling: rmvnorm()
			Multinomial Distribution: dmultinom()
________________________________________
	Hypothesis Testing and Inference
 ________________________________________
		Parametric Tests:
				t-Test:
				One-sample t-test: t.test()
				Two-sample t-test: t.test()
				Paired t-test: t.test(paired = TRUE)
				Analysis of Variance (ANOVA):
				One-Way ANOVA: aov()
				Two-Way ANOVA: aov()
				Repeated Measures ANOVA: ez::ezANOVA()
				F-Test: var.test()
				Z-Test: BSDA::z.test()
				Chi-Square Test:
				Goodness-of-fit: chisq.test()
				Test of independence: chisq.test()
________________________________________
	 Non-Parametric Tests:
				Wilcoxon Signed-Rank Test: wilcox.test()
				Mann-Whitney U Test: wilcox.test()
				Kruskal-Wallis Test: kruskal.test()
				Fisher’s Exact Test: fisher.test()
				Kolmogorov-Smirnov Test: ks.test()
________________________________________
			Confidence Intervals:
				For Mean: t.test() (for one-sample or two-sample)
				For Proportions: prop.test()
				For Variance: var.test()
				For Regression Parameters: confint()
________________________________________
	Regression and Predictive Modeling
________________________________________
		Linear Regression:
				Simple Linear Regression: lm()
				Multiple Linear Regression: lm()
				Ridge Regression: glmnet::cv.glmnet()
				Lasso Regression: glmnet::cv.glmnet()
				ElasticNet Regression: glmnet::cv.glmnet()
		________________________________________
			Generalized Linear Models:
				Logistic Regression: glm(family = binomial())
				Poisson Regression: glm(family = poisson())
				Gamma Regression: glm(family = Gamma())
				Negative Binomial Regression: MASS::glm.nb()
________________________________________
			Non-Linear Models:
				Polynomial Regression: lm() with poly()
				Support Vector Regression: e1071::svm()
				Decision Trees: rpart::rpart()
				Random Forest: randomForest::randomForest()
				Boosted Trees: xgboost::xgboost()
				K-Nearest Neighbors: class::knn()
________________________________________
	Time Series Analysis
________________________________________
				ARIMA Models:
				ARIMA: stats::arima(), forecast::auto.arima()
				Seasonal ARIMA (SARIMA): forecast::auto.arima()
				Exponential Smoothing: forecast::ets()
				State-Space Models:
				Kalman Filter: KFAS::SSModel()
				Structural Time Series: stats::decompose()
________________________________________
	Machine Learning in R
 ________________________________________
			Supervised Learning:
				Classification:
					Logistic Regression: glm()
					Random Forest: randomForest::randomForest()
					Support Vector Machines: e1071::svm()
					k-Nearest Neighbors: class::knn()
					Naive Bayes: e1071::naiveBayes()
					Regression:
					Linear Regression: lm()
					Ridge and Lasso Regression: glmnet::cv.glmnet()
________________________________________
			Unsupervised Learning:
				Clustering:
					K-Means: stats::kmeans()
					Hierarchical Clustering: stats::hclust()
					DBSCAN: dbscan::dbscan()

________________________________________
			Dimensionality Reduction:
					PCA: prcomp()
					t-SNE: Rtsne::Rtsne()
					UMAP: umap::umap()
					Deep Learning:
					Neural Networks: nnet::neuralnet()
					Convolutional Networks: keras::keras_model_sequential()
					Recurrent Neural Networks: keras::keras_model_sequential()
