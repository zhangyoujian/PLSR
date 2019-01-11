#include "PLSR.h"

PLSR::PLSR(const cube& X, const mat& Y, int item, QDialog *parent):
	X(X),Y(Y), currentItem(item),QObject(parent)
{
	if (X.n_slices != Y.n_rows)
	{
		QMessageBox::warning(NULL, "Error", "spectrum X not match concentration Y");
		return;
	}

	n_Sample = X.n_slices;
	varsX = X.slice(0).n_cols;
	varsY = 1;
	
	xmu.clear();
	xsigma.clear();
	ymu.clear();
	ysigma.clear();
	
	if(xmlfile==NULL)
	xmlfile = new QFile();
}

PLSR::PLSR(const mat X, const mat& Y, int item, QDialog *parent)
	:X(cube(0, 0, 0)),Y(Y), currentItem(item), QObject(parent)
{
	n_Sample = X.n_rows;
	traingX = X;
	
	traingY = Y.col(item);
	varsY = 1;

	xmu = arma::mean(traingX, 0);
	xsigma = arma::stddev(traingX, 0);

	ymu = arma::mean(traingY, 0);
	ysigma = arma::stddev(traingY, 0);
	if (xmlfile == NULL)
		xmlfile = new QFile();
}

PLSR::PLSR(int item,QDialog *parent):
	X(cube(0, 0, 0)), Y(Null), currentItem(item),QObject(parent)
	
{
	n_Sample = 0;
	xmu.clear();
	xsigma.clear();
	ymu.clear();
	ysigma.clear();
	varsY = 1;
	if (xmlfile == NULL)
		xmlfile = new QFile();
}

bool PLSR::createMode(int inputSize, const int comp, const double tolerance)
{
	this->tolerance = tolerance;
	if (comp < 1)
		return false;
	components = comp;
	varsX = inputSize;
	components = min(varsX-1, components);

	return true;
}

void PLSR::PLSRegression(const mat X, const mat Y, int comp)
{
	patterns = X.n_rows; // Patterns 
	varsX = X.n_cols; // X variables
	varsY = 1; // Y variables

	if (patterns != Y.n_rows) {
		cout << "The number of Predictors (X) does not match the number of Observations (Y)" << endl;
		exit(0);
	}

	if (Y.n_cols != 1) {
		cout << "PLS1 works with the case when  Y is a vector" << endl;
		exit(0);
	}

	// Check the number of components
	if (comp == -1)
		comp = components;
	//ComponentCheck(comp);
	// Score and Loading vectors used in algorithm
	vec t(patterns, fill::zeros);
	vec q(varsY, fill::zeros);
	vec p(varsX, fill::zeros);
	vec w(varsX, fill::zeros);

	// Initialize Loadings and Scores
	U = zeros<mat>(patterns, comp);
	T = zeros<mat>(patterns, comp);
	P = zeros<mat>(varsX, comp);
	Q = zeros<mat>(varsY, comp);
	W = zeros<mat>(varsX, comp);


	mat E = X; // We dont want to change X permanently
	mat F = Y; // We dont want to change Y permanently


			   // Beggining of the algorithm
	w = normalise(X.t()*Y); // Initial estimate of w
	t = X*w; // First score of X

			 // Find comp number of components
	for (int i = 0; i < comp; i++) {
		// The algorithm stops when comp components found or when E is a null matrix.
		double tk = conv_to<double>::from(t.t()*t); // Squared L2 norm of t
		t /= tk; // divide t by tk
		p = E.t()*t; // jth loading of X
		q = Y.t()*t; // jth loading of Y

		if (norm(q) == 0)break;
		// Update PLS components
		W.col(i) = w;
		P.col(i) = p;
		Q.col(i) = q;
		T.col(i) = t;

		E -= tk*t*p.t(); // Deflation of E
						 // If E become a "null" matrix, stop iteration 
		if (all(vectorise(E) < tolerance))
			break;

		w = E.t()*Y; // jth + weight
		t = E*w; // jth +1 score of X
	} // End of regression loop

}




int PLSR::getBestComp()
{
	return minEMSECVCom;
}

const mat PLSR::getStatistic()
{
	return StatisticData;
}


//================================================public API===========================================

mat PLSR::VarExp(const mat X, const mat Y, const int comp )
{
	return 1 - (SSE(X, Y, comp) / TSS(Y));
}

mat PLSR::Residuals(const mat X, const mat Y, int comp )
{
	// Check the number of components
	if (comp == -1) comp = components;
	//ComponentCheck(min(patterns,varsX), comp);
	// Return the errors (Residual Space)
	return Y - FittedValues(X, comp);
}

rowvec PLSR::getR2()
{
	return R2;
}

mat PLSR::FittedValues(const mat X, int comp )
{
	// Check the number of components
	if (comp == -1) comp = components;
	//ComponentCheck(min(patterns,varsX), comp);
	// Return the Fitted Values
	mat normaBeta = Coefficients(comp);
	return plsnorma(X)*normaBeta*ysigma(0) + ymu(0);
}


rowvec PLSR::TSS(const mat Y)
{
	rowvec tss(varsY, fill::zeros); // TSS values
							// For every variable of Y calculate the TSS
	for (int i = 0; i < varsY; i++)
		tss.col(i) = sum(pow((Y.col(i) - mean(Y.col(i))), 2)); // TSS if ith variable.
	return tss;
}

rowvec PLSR::SSE(const mat X, const mat Y, const int comp )
{
	mat res = Residuals(X, Y, comp); // Calculate the residuals
	rowvec e(Y.n_cols, fill::zeros);
	// For every variable of y calculate the SSE
	for (int i = 0; i< varsY; i++)
		e.col(i) = res.col(i).t()*res.col(i); // SSE of ith variable.
	return e;
}

mat PLSR::LOOCV_Residuals(int item,const int comp)
{
	vec T = Y.col(item);
	mat res(n_Sample, comp); // Residuals
	cube Xtr = X.slices(1, n_Sample - 1);
	vec Ytr = T.rows(1, n_Sample - 1);
	emit sendCurrentProcess(0, n_Sample*comp);
	for (register int i = 0; i < n_Sample; i++)
	{
		convertTomat(Xtr, Ytr);
		PLSRegression(zscore(traingX), zscore(traingY), comp); // Train
					// For every number of components calculate the residuals
		for (register int j = 0; j< comp; j++) 
		{
			mat yfitpls = FittedValues(X.slice(i), j + 1);
			yfitpls = sort(yfitpls,"ascend");
			yfitpls = mean(yfitpls.rows(1, yfitpls.n_rows - 2), 0);
			res(i, j) = T(i)- yfitpls(0,0);
			emit sendCurrentProcess(i*comp + (j + 1), n_Sample*comp);
		} // End of Residuals for

		  // If cross validation finished continue. 
		if (i < (int)Xtr.n_slices) {
			Xtr.slice(i) = X.slice(i);
			Ytr(i) = T(i);
		}

	} // End of cross validation for
	return res;
}

mat PLSR::LOOCV_Residuals_Single(int item, const int comp )
{
	mat res = zeros(n_Sample,  comp);  // Residuals
	mat Xtr = traingX.rows(1, n_Sample - 1); // Trainning Observations 
	mat Ytr = traingY.rows(1, n_Sample - 1); // Trainning Predictions
									   // Validate the residuals for every pattern and every comp combination
	for (register int i = 0; i < n_Sample; i++)
	{
		xmu = arma::mean(Xtr, 0);
		xsigma = arma::stddev(Xtr, 0);
		ymu = arma::mean(Ytr, 0);
		ysigma = arma::stddev(Ytr, 0);

		PLSRegression(zscore(Xtr), zscore(Ytr), comp); // Train
					// For every number of components calculate the residuals
		for (register int j = 0; j< comp; j++) {

			mat yfitpls = FittedValues(traingX.row(i), j + 1);
			res(i, j) = traingY(i,0) - yfitpls(0, 0);
			emit sendCurrentProcess(i*comp + (j + 1), n_Sample*comp);
		} // End of Residuals for

		  // If cross validation finished continue. 
		if (i < (int)Xtr.n_rows) {
			Xtr.row(i) = traingX.row(i);
			Ytr.row(i) = traingY.row(i);
		}

	} // End of cross validation for
	return res;
}




mat PLSR::Coefficients( int comp )
{
	if (comp == -1) comp = components;
	//ComponentCheck(varsX, comp);

	mat tem = P.t()*W; 
	//tem = pinv(tem); // Moore-Penrose Pseudo inverse of temp
	tem = inv(tem);
	return W.cols(0, comp - 1)*tem.rows(0, comp - 1).cols(0, comp - 1)*Q.cols(0, comp - 1).t(); // Coefficients
}


void PLSR::LOOCV()
{
	// Check the number of components
	int  comp = components;
	int item = currentItem;
	mat res;
	time.start();
	//ComponentCheck(min(patterns,varsX), comp);
	if (X.is_empty())
	{
		res = LOOCV_Residuals_Single(item, comp);
	}
	else
	{
		res = LOOCV_Residuals(item, comp); // Acquire the Residuals
	}

	vec y = Y.col(item);
	mat statistics(comp,4, fill::zeros); // Statistics
	vec SSE(comp,fill::zeros); // Sum of Squared Errors
	vec RMSE(comp,fill::zeros); // Root Mean Squared Errors
	vec MSE(comp, fill::zeros); // Mean Squared Errors
	vec RSquare(comp, fill::zeros); // R Squared
									  // For every number of components
	for (int i = 0; i < comp; i++)
	 {
			// Calculate the SSE and R2
		vec temp = res.col(i);
		SSE(i) = dot(temp, temp);
		RSquare(i) = 1 - SSE(i) / dot(y.t(), y);
		}
	MSE = SSE / n_Sample;
	RMSE = sqrt(MSE);

	// Statistics
	statistics.col(0) = SSE;
	statistics.col(1) = MSE;
	statistics.col(2) = RMSE;
	statistics.col(3) = RSquare;

	uvec indices = sort_index(RMSE,"ascend");
	minEMSECVCom = indices(0)+1;   //确定最终的主成分数
	StatisticData.clear();
	StatisticData = statistics;


	//=====================================
	xmu = arma::mean(traingX, 0);
	xsigma = arma::stddev(traingX, 0);
	ymu = arma::mean(traingY, 0);
	ysigma = arma::stddev(traingY, 0);

	PLSRegression(zscore(traingX), zscore(traingY), comp); // Train
	R2.clear();
	R2 = rowvec(comp, fill::zeros);

	for (int i = 1; i <= comp; i++)
	{
		mat qsuare = VarExp(traingX, traingY, i);
		R2(i - 1) = qsuare(0, 0);
	}
	double toc = time.elapsed() / 1000.0;
	emit sendtimeLapse(toc);
	emit SendTrainingFinished();
	//return StatisticData;
}


mat PLSR::Predict(const mat X, int comp)
{

	if (comp == -1)
		comp = minEMSECVCom;
	else if (comp > components)
		comp = components;
	else
	{
	}
	
	return FittedValues(X, comp);
}


bool PLSR::convertTomat(const cube xdata, const vec T)
{
	if (xdata.is_empty())
		return false;
	
	traingX.clear();
	traingY.clear();
	traingX = xdata.slice(0);
	int count_per = traingX.n_rows;
	traingY = T(0)*ones(count_per, 1);
	int num = xdata.n_slices;
	for (int i = 1; i < num; i++)
	{
		count_per = xdata.slice(i).n_rows;
		traingX = join_cols(traingX, xdata.slice(i));
		traingY = join_cols(traingY, T(i)*ones(count_per, 1));
	}

	xmu = arma::mean(traingX, 0);
	xsigma = arma::stddev(traingX, 0);

	ymu = arma::mean(traingY, 0);
	ysigma = arma::stddev(traingY, 0);

	return true;
}


PLSR::~PLSR()
{
	if (xmlfile)
		delete xmlfile;

}

//=============================光谱标准化================================================================
mat PLSR::zscore(const mat M)
{
	mat mu = arma::mean(M, 0);
	mat sigma = arma::stddev(M, 0);
	mat res = (M - repmat(mu, M.n_rows, 1)) / (repmat(sigma, M.n_rows, 1));
	if (res.has_inf())
		res.replace(datum::inf, 0);
	if (res.has_nan())
		res.replace(datum::nan, 0);
	return res;
}
mat PLSR::plsnorma(const mat X)
{
	mat res = (X - repmat(xmu, X.n_rows, 1)) / (repmat(xsigma, X.n_rows, 1));
	if (res.has_inf())
		res.replace(datum::inf, 0);
	if (res.has_nan())
		res.replace(datum::nan, 0);
	return res;
}

//===========================================模型的保存与载入=================================================
bool PLSR::save()
{
	QDomDocument doc;
	QDomProcessingInstruction instruction = doc.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"UTF-8\"");
	doc.appendChild(instruction);  //添加标题
								   //添加结点
	QDomElement Nodes = doc.createElement("PLSR_storage");
	doc.appendChild(Nodes);
	QDomElement parNode = doc.createElement("parameter");

	QDomElement ItemNode = doc.createElement("Item");
	QDomText itemData = doc.createTextNode(QString::number(currentItem));
	ItemNode.appendChild(itemData);
	parNode.appendChild(ItemNode);

	QDomElement MaxCompNode = doc.createElement("maxComponent");
	QDomText MaxCompData = doc.createTextNode(QString::number(components));
	MaxCompNode.appendChild(MaxCompData);
	parNode.appendChild(MaxCompNode);

	QDomElement varsXNode = doc.createElement("varsX");
	QDomText varsXData = doc.createTextNode(QString::number(varsX));
	varsXNode.appendChild(varsXData);
	parNode.appendChild(varsXNode);

	QDomElement toleranceNode = doc.createElement("tolerance");
	QDomText toleranceData = doc.createTextNode(QString::number(tolerance));
	toleranceNode.appendChild(toleranceData);
	parNode.appendChild(toleranceNode);

	QDomElement bestCompNode = doc.createElement("minEMSECVCom");
	QDomText bestCompData = doc.createTextNode(QString::number(minEMSECVCom));
	bestCompNode.appendChild(bestCompData);
	parNode.appendChild(bestCompNode);
	Nodes.appendChild(parNode);


	//=======================================保存下一个结点信息============================
	QString datasequnce;
	QDomElement NormalNode = doc.createElement("NormaLise");
	QDomElement xmuNode = doc.createElement("xmu");
	datasequnce.clear();
	for (int j = 0; j < xmu.n_cols; j++)
	{
		if (j == xmu.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(xmu(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(xmu(j)) + " ";
	}
	QDomText xmudata = doc.createTextNode(datasequnce);
	xmuNode.appendChild(xmudata);
	NormalNode.appendChild(xmuNode);


	QDomElement xsigmaNode = doc.createElement("xsigma");
	datasequnce.clear();
	for (int j = 0; j < xsigma.n_cols; j++)
	{
		if (j == xsigma.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(xsigma(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(xsigma(j)) + " ";
	}
	QDomText xsigmadata = doc.createTextNode(datasequnce);
	xsigmaNode.appendChild(xsigmadata);
	NormalNode.appendChild(xsigmaNode);


	QDomElement ymuNode = doc.createElement("ymu");
	datasequnce.clear();
	for (int j = 0; j < ymu.n_cols; j++)
	{
		if (j == ymu.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(ymu(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(ymu(j)) + " ";
	}
	QDomText ymudata = doc.createTextNode(datasequnce);
	ymuNode.appendChild(ymudata);
	NormalNode.appendChild(ymuNode);


	QDomElement ysigmaNode = doc.createElement("ysigma");
	datasequnce.clear();
	for (int j = 0; j < ysigma.n_cols; j++)
	{
		if (j == ysigma.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(ysigma(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(ysigma(j)) + " ";
	}
	QDomText ysigmadata = doc.createTextNode(datasequnce);
	ysigmaNode.appendChild(ysigmadata);
	NormalNode.appendChild(ysigmaNode);
	Nodes.appendChild(NormalNode);


	//=========================================开始创建统计结点============================================
	QDomElement ModeInfoNode = doc.createElement("ModeInfo");
	QDomElement R2Node = doc.createElement("R2");
	datasequnce.clear();
	for (int j = 0; j < R2.n_cols; j++)
	{
		if (j == R2.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(R2(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(R2(j)) + " ";
	}
	QDomText R2data = doc.createTextNode(datasequnce);
	R2Node.appendChild(R2data);
	ModeInfoNode.appendChild(R2Node);


	QDomElement StatisticNode = doc.createElement("Statistic");
	rowvec S = reshape(StatisticData, 1, StatisticData.n_elem);
	datasequnce.clear();
	for (int j = 0; j < S.n_cols; j++)
	{
		if (j == S.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(S(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(S(j)) + " ";
	}
	QDomText Sdata = doc.createTextNode(datasequnce);
	StatisticNode.appendChild(Sdata);
	ModeInfoNode.appendChild(StatisticNode);
	Nodes.appendChild(ModeInfoNode);

	//=======================================添加权重信息结点=============================
	QDomElement WeightNode = doc.createElement("weight");

	QDomElement WNode = doc.createElement("W");
	rowvec wvec = reshape(W, 1, W.n_elem);
	datasequnce.clear();
	for (int j = 0; j < wvec.n_cols; j++)
	{
		if (j == wvec.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(wvec(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(wvec(j)) + " ";
	}
	QDomText Wdata = doc.createTextNode(datasequnce);
	WNode.appendChild(Wdata);
	WeightNode.appendChild(WNode);

	QDomElement PNode = doc.createElement("P");
	rowvec pvec = reshape(P, 1, P.n_elem);
	datasequnce.clear();
	for (int j = 0; j < pvec.n_cols; j++)
	{
		if (j == pvec.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(pvec(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(pvec(j)) + " ";
	}
	QDomText Pdata = doc.createTextNode(datasequnce);
	PNode.appendChild(Pdata);
	WeightNode.appendChild(PNode);

	QDomElement QNode = doc.createElement("Q");
	rowvec qvec = reshape(Q, 1, Q.n_elem);
	datasequnce.clear();
	for (int j = 0; j < qvec.n_cols; j++)
	{
		if (j == qvec.n_cols - 1)
		{
			datasequnce = datasequnce + QString::number(qvec(j)) + "\0";
		}
		else
			datasequnce = datasequnce + QString::number(qvec(j)) + " ";
	}
	QDomText qdata = doc.createTextNode(datasequnce);
	QNode.appendChild(qdata);
	WeightNode.appendChild(QNode);
	Nodes.appendChild(WeightNode);

	QString Path;
	extern QString PLSModePath;
	Path = PLSModePath + QString("PLSR_DATA_ITEM%1.xml").arg(currentItem);

	xmlfile->setFileName(Path);
	bool ret = xmlfile->open(QIODevice::WriteOnly | QIODevice::Text);
	if (!ret)
	{
		QMessageBox::warning(NULL, "warning", "save mode file failed");
		xmlfile->close();
		return false;
	}

	QTextStream stream(xmlfile);
	stream.setCodec("UTF-8");
	doc.save(stream, 4);  //文件每行缩进4个空格
	xmlfile->close();
	return true;
}
bool PLSR::load()
{
	QString Path;
	extern QString PLSModePath;
	Path = PLSModePath + QString("PLSR_DATA_ITEM%1.xml").arg(currentItem);
	xmlfile->setFileName(Path);
	bool ret = xmlfile->open(QIODevice::ReadOnly | QIODevice::Text);
	if (!ret)
	{
		xmlfile->close();
		QMessageBox::warning(NULL, "warning", "mode load failed!");
		return false;
	}
	QDomDocument doc;
	ret = doc.setContent(xmlfile);
	if (!ret)
	{
		xmlfile->close();
		QMessageBox::warning(NULL, "warning", "关联xml文件失败");
		return false;
	}
	xmlfile->close();
	QString datasequnce;
	QDomElement docElm = doc.documentElement();
	QDomNode n = docElm.firstChild();   //partype

	if (n.nodeName() != "parameter")
		return false;

	QDomNodeList childList = n.childNodes();
	for (int i = 0; i < childList.length(); i++)
	{
		QDomNode node = childList.at(i);
		QDomElement childElement = node.toElement();
		datasequnce.clear();
		datasequnce = childElement.text();
		if (node.nodeName() == "Item")
		{
			if (datasequnce.toInt() != currentItem)
				return false;
			currentItem = datasequnce.toInt();
		}
		else if (node.nodeName() == "maxComponent")
		{
			components = datasequnce.toInt();
		}
		else if (node.nodeName() == "varsX")
		{
			varsX = datasequnce.toInt();
		}
		else if (node.nodeName() == "tolerance")
		{
			tolerance = datasequnce.toDouble();
		}
		else if (node.nodeName() == "minEMSECVCom")
		{
			minEMSECVCom = datasequnce.toInt();
		}
		else {}

	}
	//============================解析下一个结点========================================
	n = n.nextSibling();   //partype
	if (n.nodeName() != "NormaLise")
		return false;
	childList = n.childNodes();
	for (int i = 0; i < childList.length(); i++)
	{
		QDomNode node = childList.at(i);
		QDomElement childElement = node.toElement();
		datasequnce.clear();
		datasequnce = childElement.text();
		QStringList splitdata = datasequnce.split(' ');
		if (node.nodeName() == "xmu")
		{
			xmu = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				xmu(k) = splitdata.at(k).toDouble();
			}
		}
		else if(node.nodeName() == "xsigma")
		{
			xsigma = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				xsigma(k) = splitdata.at(k).toDouble();
			}
		}
		else if (node.nodeName() == "ymu")
		{
			ymu = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				ymu(k) = splitdata.at(k).toDouble();
			}
		}
		else if (node.nodeName() == "ysigma")
		{
			ysigma = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				ysigma(k) = splitdata.at(k).toDouble();
			}
		}
		else {}

	}

	//============================解析下一个结点========================================
	n = n.nextSibling();   
	if (n.nodeName() != "ModeInfo")
		return false;
	childList = n.childNodes();
	for (int i = 0; i < childList.length(); i++)
	{
		QDomNode node = childList.at(i);
		QDomElement childElement = node.toElement();
		datasequnce.clear();
		datasequnce = childElement.text();
		QStringList splitdata = datasequnce.split(' ');
		if (node.nodeName() == "R2")
		{
			R2 = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				R2(k) = splitdata.at(k).toDouble();
			}
		}
		else if (node.nodeName() == "Statistic")
		{
			rowvec weight = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				weight(k) = splitdata.at(k).toDouble();
			}
			StatisticData = reshape(weight,components,4);
		}
		else {}
	}

	//================================解析下一个结点==============================
	n = n.nextSibling();
	if (n.nodeName() != "weight")
		return false;
	childList = n.childNodes();
	for (int i = 0; i < childList.length(); i++)
	{
		QDomNode node = childList.at(i);
		QDomElement childElement = node.toElement();
		datasequnce.clear();
		datasequnce = childElement.text();
		QStringList splitdata = datasequnce.split(' ');
		if (node.nodeName() == "W")
		{
			rowvec w= rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				w(k) = splitdata.at(k).toDouble();
			}
			W = reshape(w, varsX, components);
		}
		else if (node.nodeName() == "P")
		{
			rowvec p = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				p(k) = splitdata.at(k).toDouble();
			}
			P = reshape(p, varsX, components);
		}
		else if (node.nodeName() == "Q")
		{
			rowvec q = rowvec(splitdata.size(), fill::zeros);
			for (int k = 0; k < splitdata.size(); k++)
			{
				q(k) = splitdata.at(k).toDouble();
			}
			Q = reshape(q, varsY, components);
		}
		else {}
	}

	if (!this->createMode(varsX, components, tolerance))
		return false;
	return true;
}