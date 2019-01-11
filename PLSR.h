#pragma once
#include"stdafx.h"
#include"qobject.h"
#include"qdialog.h"
#include"QTime" 

class PLSR: public QObject
{
	Q_OBJECT
public:
	PLSR(const cube& X, const mat& Y, int item,  QDialog *parent =0);
	PLSR(const mat X, const mat& Y, int item,QDialog *parent = 0);
	PLSR(int item, QDialog *parent = 0);
	~PLSR();
	bool createMode(int inputSize,const int comp = -1, const double tolerance = 0.00001);
	
	mat Predict(const mat X, int comp=-1);
	bool save();
	bool load();

	int getBestComp();
	const mat getStatistic();
	rowvec getR2();
public slots:
	void train() { LOOCV(); };


signals:
	void sendCurrentProcess(int, int);
	void SendTrainingFinished();
	void sendtimeLapse(double mssecond);

private:
	mat FittedValues(const mat X, int comp = -1);       //预测
	void PLSRegression(const mat X, const mat Y, int comp = -1);
	
	
	void LOOCV();
	mat LOOCV_Residuals(int item, const int comp = -1);
	mat LOOCV_Residuals_Single(int item, const int comp = -1);

protected:
	//! Used to initialize Observations and Predictions.
   mat Null;

   //训练集数据
   mat traingX;
   mat traingY;


  //! Observations.
  const cube &X;
  //! Predictions.
  const mat &Y;

  //! Number of components to use (By default is 10)
  int components;

  //! The latent vectors or score matrix of X.
  mat T;

  //! The loadings of X.
  mat P;

  //! The score matrix of Y.
  mat U;

  //! The weight matrix or the loadings of Y.
  mat Q;

  //! Weights
  mat W;

  //! The number of patterns (data)
  int patterns;
  int n_Sample;
  //! The numver of X-variables
  int varsX;

  //! The number of Y-variables
  int varsY; 

  //! The tolerance for terminationx
  double tolerance;


private:

	mat Residuals(const mat X, const mat Y, int comp = -1);

	rowvec SSE(const mat X, const mat Y, const int comp = -1);
	rowvec TSS(const mat Y);
	mat VarExp(const mat X, const mat Y, const int comp = -1);
	mat Coefficients(const int comp = -1);
	
	
private :
	//for trainingData function
	
//===================光谱预处理函数============================
	mat plsnorma(const mat X);
	bool convertTomat(const cube xdata, const vec target);
	mat zscore(const mat M);
	double& Tolerance() { return tolerance; };
private:
	QFile *xmlfile = NULL;
	QTime time;

	

	
	int currentItem;
	int minEMSECVCom;
	rowvec xmu;
	rowvec xsigma;
	rowvec ymu;
	rowvec ysigma;
	rowvec R2;
	mat StatisticData;

};

