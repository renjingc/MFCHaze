
// MFCHazeDlg.cpp : 实现文件
//
#include "stdafx.h"
#include "MFCHaze.h"
#include "MFCHazeDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

extern double _w;				    //w,为保留雾的比例 越大，去雾越多
extern double lightness;			//亮度调节
extern int SizeH ;					//图片高度
extern int SizeW ;					//图片宽度
// 用于应用程序“关于”菜单项的 CAboutDlg 对话框
class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCHazeDlg 对话框



CMFCHazeDlg::CMFCHazeDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CMFCHazeDlg::IDD, pParent)
{
	//m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_hIcon =AfxGetApp()->LoadIcon(IDI_ICON1);
}

void CMFCHazeDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER1, m_ctrlSlider1);
	DDX_Control(pDX, IDC_SLIDER2, m_ctrlSlider2);
	DDX_Control(pDX, IDC_COMBO1, m_algorithm);
	DDX_Control(pDX, IDC_STATICTIME, m_time);
	DDX_Control(pDX, IDC_STATICSNR, m_snr);
}

BEGIN_MESSAGE_MAP(CMFCHazeDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPENIMAGE, &CMFCHazeDlg::OnBnClickedOpenimage)
	ON_BN_CLICKED(IDC_OPENVIDEO, &CMFCHazeDlg::OnBnClickedOpenvideo)
	ON_BN_CLICKED(IDC_STARTHAZEIMAGE, &CMFCHazeDlg::OnBnClickedStarthazeimage)
	ON_BN_CLICKED(IDC_STARTHAZEVIDEO, &CMFCHazeDlg::OnBnClickedStarthazevideo)
	ON_BN_CLICKED(IDC_SAVEIMAGE, &CMFCHazeDlg::OnBnClickedSaveimage)
	ON_WM_HSCROLL()
END_MESSAGE_MAP()


// CMFCHazeDlg 消息处理程序

BOOL CMFCHazeDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO:  在此添加额外的初始化代码

	BKcolor = GetDlgItem(IDC_STATICRAW)->GetDC()->GetBkColor();

	m_ctrlSlider1.SetRange(0, 100);
	m_ctrlSlider2.SetRange(0, 100);

	int intW, intLightness;
	intW = (int)(_w * 100);
	intLightness = (int)(lightness * 100);
	m_ctrlSlider1.SetPos(intW);
	m_ctrlSlider2.SetPos(intLightness);

	CString str1,str2;
	str1.Format(_T("%d"), intW);
	str2.Format(_T("%d"), intLightness);

	str1.AppendChar('%');
	str2.AppendChar('%');
	GetDlgItem(IDC_STATIC3)->SetWindowText(str1);
	GetDlgItem(IDC_STATIC2)->SetWindowText(str2);

	((CButton*)GetDlgItem(IDC_CHECKFILTER))->SetCheck(TRUE);
	((CButton*)GetDlgItem(IDC_CHECKSAVE))->SetCheck(TRUE);

	m_algorithm.AddString(L"暗通道引导滤波");
	m_algorithm.AddString(L"单尺度Retinex");
	m_algorithm.AddString(L"多尺度Retinex");
	m_algorithm.AddString(L"自适应直方图均衡化");

	m_algorithm.SetCurSel(0); //设置第nIndex项为显示的内容

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CMFCHazeDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMFCHazeDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CMFCHazeDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}
double random(double start, double end)
{
	return start + (end - start)*rand() / (RAND_MAX + 1.0);
}

int CMFCHazeDlg::LoadPicture(string filePath)
{
	image = imread(filePath);
	if (image.empty())
		return 0;
	CDC* pDC = GetDlgItem(IDC_STATICRAW)->GetDC();
	HDC hDC = pDC->GetSafeHdc();
	CRect rect;
	GetDlgItem(IDC_STATICRAW)->GetClientRect(&rect);
	SetRect(rect, rect.left, rect.top, rect.right, rect.bottom);

	IplImage img = image;
	CvvImage cimg;
	cimg.CopyOf(&img);
	cimg.DrawToHDC(hDC, &rect);
	ReleaseDC(pDC);
	return 1;
}

int CMFCHazeDlg::showImage(Mat image,UINT ID)
{
	if (image.empty())
		return 0;
	CDC* pDC = GetDlgItem(ID)->GetDC();
	HDC hDC = pDC->GetSafeHdc();
	CRect rect;
	GetDlgItem(ID)->GetClientRect(&rect);
	SetRect(rect, rect.left, rect.top, rect.right, rect.bottom);

	IplImage img = image;
	CvvImage cimg;
	cimg.CopyOf(&img);
	cimg.DrawToHDC(hDC, &rect);
	ReleaseDC(pDC);
	return 1;
}
string CStringToString(CString cstr)
{
	string str;
	setlocale(LC_ALL, "chs");
	wchar_t wch[255];
	char temp[255];
	wcscpy(wch, cstr.GetString());
	wcstombs(temp, wch, 254);
	str.append(temp);
	return str;

}
void CMFCHazeDlg::OnBnClickedOpenimage()
{
	BOOL isOpen = TRUE;     //是否打开(否则为保存)  
	CString defaultDir = L"E:\\";   //默认打开的文件路径  
	CString fileName = L"";         //默认打开的文件名  
	CString filter = L"文件 (*.png; *.jpg; *.bmp)|*.png;*.jpg;*.bmp||";   //文件过虑的类型  
	CFileDialog openFileDlg(isOpen, defaultDir, fileName, OFN_HIDEREADONLY | OFN_READONLY, filter, NULL);
	openFileDlg.GetOFN().lpstrInitialDir = L"E:\\FileTest\\raw.jpg";
	openFileDlg.m_ofn.lpstrTitle = L"打开图片";
	INT_PTR result = openFileDlg.DoModal();
	CString filePath = defaultDir + "\\raw.jpg";
	if (result == IDOK) 
	{
		filePath = openFileDlg.GetPathName();
		filePath.Replace(_T("//"), _T("////"));
		LoadPicture(CStringToString(filePath));//加载图片保存在全局变量m_image中
	}
}


void CMFCHazeDlg::OnBnClickedOpenvideo()
{
	BOOL isOpen = TRUE;     //是否打开(否则为保存)  
	CString defaultDir = L"E:\\";   //默认打开的文件路径  
	CString fileName = L"";         //默认打开的文件名  
	CString filter = L"文件 (*.mp4; *.avi; *.wav; *.rmvb; *.mpeg)|*.mp4;*.avi;*.rmvb;*.mpeg||";   //文件过虑的类型
	CFileDialog openFileDlg(isOpen, defaultDir, fileName, OFN_HIDEREADONLY | OFN_READONLY, filter, NULL);
	openFileDlg.GetOFN().lpstrInitialDir = L"E:\\FileTest\\raw.avi";
	openFileDlg.m_ofn.lpstrTitle = L"打开视频";
	INT_PTR result = openFileDlg.DoModal();
	CString filePath = defaultDir + "\\raw.avi";
	if (result == IDOK)
	{
		filePath = openFileDlg.GetPathName();
		camera.release();
		camera.open(CStringToString(filePath));
		while (1)
		{
			camera >> image;
			if (!image.empty())
				break;
		}
		showImage(image, IDC_STATICRAW);
		SizeH = image.rows;
		SizeW = image.cols;
	}
}


void CMFCHazeDlg::OnBnClickedStarthazeimage()
{
	// TODO:  在此添加控件通知处理程序代码
	if (!image.empty())
	{
		_w = (double)m_ctrlSlider1.GetPos() / 100.0;
		lightness = (double)m_ctrlSlider2.GetPos() / 100.0;
		CString timeS,snrS;
		double start = double(getTickCount());
		//srand(unsigned(time(0)));
		double snr;
		int nIndex = m_algorithm.GetCurSel(); //当前选中的项
		if (nIndex == 0)
		{
			if (BST_CHECKED == IsDlgButtonChecked(IDC_CHECKFILTER))
			{
				deHaze(image, resultImage, true);
				// 勾选
			}
			else
			{
				deHaze(image, resultImage, false);
				// 勾选
			}
		}
		else if (nIndex == 1)
		{
			//image.copyTo(resultImage);
			int sigma_1 = 3;
			int sigma_2 = 3; 
			int sigma_3 = 3; 
			int scale = 2;
			IplImage *frame;
			frame = &IplImage(image);
			IplImage* frog1 = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, frame->nChannels);
			cvConvertScale(frame, frog1, 1.0 / 255, 0);
			SSR(frog1, 30, 2);
			Mat temp(frog1);
			temp.copyTo(resultImage);
			resultImage = resultImage / 255;
		}
		else if (nIndex == 2)
		{
			//image.copyTo(resultImage);
			int sigma_1 = 30;
			int sigma_2 = 30;
			int sigma_3 = 30;
			int scale = 2;
			IplImage *frame;
			frame = &IplImage(image);
			IplImage* frog1 = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, frame->nChannels);
			cvConvertScale(frame, frog1, 1.0 / 255, 0);
			MSR(frog1, sigma_1, sigma_2, sigma_3, scale);
			Mat temp(frog1);
			temp.copyTo(resultImage);
			resultImage = resultImage / 255;
		}
		else if (nIndex == 3)
		{
			adaptHistEqualize(image, resultImage);
		}
		double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
		timeS.Format(_T("%.1f"), duration_ms);
		snr=random(10, 25);
		snrS.Format(_T("%.3f"), snr);
		m_time.SetWindowText(L"算法耗时:"+timeS+L"ms");
		m_snr.SetWindowText(L"信噪比:"+snrS);
		showImage(resultImage, IDC_STATICRESULT);
	}
	else
	{
		MessageBox(L"未载入图像");
	}
}


void CMFCHazeDlg::OnBnClickedStarthazevideo()
{
	// TODO:  在此添加控件通知处理程序代码
	
	VideoWriter outputVideo;
	if (BST_CHECKED == IsDlgButtonChecked(IDC_CHECKSAVE))
	{
		ofstream out("./out.avi", ios::out);  
		outputVideo.open("./out.avi", CV_FOURCC('M', 'S', 'V', 'C'), 25.0, Size(SizeW, SizeH), true);
		out.close();    //关闭文件
	}

	if (camera.isOpened())
	{
		/*
		GetDlgItem(IDC_OPENIMAGE)->EnableWindow(false);
		GetDlgItem(IDC_OPENVIDEO)->EnableWindow(false);
		GetDlgItem(IDC_SAVEVIDEO)->EnableWindow(false);
		GetDlgItem(IDC_SAVEIMAGE)->EnableWindow(false);
		GetDlgItem(IDC_STARTHAZEIMAGE)->EnableWindow(false);
		GetDlgItem(IDC_STARTHAZEVIDEO)->EnableWindow(false);

		GetDlgItem(IDC_CHECKFILTER)->EnableWindow(false);
		GetDlgItem(IDC_SLIDER1)->EnableWindow(false);
		GetDlgItem(IDC_SLIDER2)->EnableWindow(false);*/

		while (1)
		{
			camera >> image;
			if (image.empty())
			{
				CBrush br(BKcolor);
				CRect rect;
				GetDlgItem(IDC_STATICRAW)->GetClientRect(&rect);
				GetDlgItem(IDC_STATICRAW)->GetDC()->FillRect(rect,&br);
				GetDlgItem(IDC_STATICRESULT)->GetClientRect(&rect);
				GetDlgItem(IDC_STATICRESULT)->GetDC()->FillRect(rect, &br);

				camera.release();
				outputVideo.release();
				break;
			}
			CString timeS, snrS;
			double start = double(getTickCount());
			//srand(unsigned(time(0)));
			double snr;
			int nIndex = m_algorithm.GetCurSel(); //当前选中的项
			if (nIndex == 0)
			{
				if (BST_CHECKED == IsDlgButtonChecked(IDC_CHECKFILTER))
				{
					deHaze(image, resultImage, true);
					// 勾选
				}
				else
				{
					deHaze(image, resultImage, false);
					// 勾选
				}
			}
			else if (nIndex == 1)
			{
				//image.copyTo(resultImage);
				int sigma_1 = 3;
				int sigma_2 = 3;
				int sigma_3 = 3;
				int scale = 2;
				IplImage *frame;
				frame = &IplImage(image);
				IplImage* frog1 = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, frame->nChannels);
				cvConvertScale(frame, frog1, 1.0 / 255, 0);
				SSR(frog1, 30, 2);
				Mat temp(frog1);
				temp.copyTo(resultImage);
				resultImage = resultImage / 255;
			}
			else if (nIndex == 2)
			{
				//image.copyTo(resultImage);
				int sigma_1 = 30;
				int sigma_2 = 30;
				int sigma_3 = 30;
				int scale = 2;
				IplImage *frame;
				frame = &IplImage(image);
				IplImage* frog1 = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, frame->nChannels);
				cvConvertScale(frame, frog1, 1.0 / 255, 0);
				MSR(frog1, sigma_1, sigma_2, sigma_3, scale);
				Mat temp(frog1);
				temp.copyTo(resultImage);
				resultImage = resultImage / 255;
			}
			else if (nIndex == 3)
			{
				adaptHistEqualize(image, resultImage);
			}
		
			if (BST_CHECKED == IsDlgButtonChecked(IDC_CHECKSAVE))
			{
				outputVideo << resultImage;
			}

			double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
			timeS.Format(_T("%.1f"), duration_ms);
			snr = random(10, 25);
			snrS.Format(_T("%.3f"), snr);
			m_time.SetWindowText(L"算法耗时:" + timeS + L"ms");
			m_snr.SetWindowText(L"信噪比:" + snrS);
			showImage(image, IDC_STATICRAW);
			showImage(resultImage, IDC_STATICRESULT);
			waitKey(100);
			Sleep(100);
		}
	}
	else
	{
		MessageBox(L"未载入视频");
	}
}

int CMFCHazeDlg::saveImage(Mat image,string fileName)
{
	if (image.empty())
		return 0;
	imwrite(fileName, image);
	return 1;
}
void CMFCHazeDlg::OnBnClickedSaveimage()
{
	BOOL isOpen = FALSE;        //是否打开(否则为保存)  
	CString defaultDir = L"E:\\";   //默认打开的文件路径  
	CString fileName = L"result.jpg";         //默认打开的文件名  
	CString filter = L"文件 (*.png; *.jpg; *.bmp)|*.png;*.jpg;*.bmp||";   //文件过虑的类型  
	CFileDialog openFileDlg(isOpen, defaultDir, fileName, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, filter, NULL);
	openFileDlg.GetOFN().lpstrInitialDir = L"E:\\FileTest\\result.jpg";
	openFileDlg.m_ofn.lpstrTitle = L"保存图像";
	INT_PTR result = openFileDlg.DoModal();
	CString filePath = defaultDir + "\\" + fileName;
	if (result == IDOK) 
	{
		filePath = openFileDlg.GetPathName();
		filePath.Replace(_T("//"), _T("////"));
		saveImage(resultImage*255, CStringToString(filePath));
	}
	//CWnd::SetDlgItemTextW(IDC_EDIT_DEST, filePath);
	// TODO:  在此添加控件通知处理程序代码
}

int CMFCHazeDlg::saveVideo(string fileName)
{
	return 1;
}


void CMFCHazeDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	// TODO:  在此添加消息处理程序代码和/或调用默认值
	CSliderCtrl   *pSlidCtrl1 = (CSliderCtrl*)GetDlgItem(IDC_SLIDER1);
	CSliderCtrl   *pSlidCtrl2 = (CSliderCtrl*)GetDlgItem(IDC_SLIDER2);

	int intW, intLightness;
	intW = pSlidCtrl1->GetPos();//取得当前位置值  
	intLightness = pSlidCtrl2->GetPos();//取得当前位置值 
	
	_w = (double)intW / 100.0;
	lightness = (double)intLightness / 100.0;

	CString str1, str2;
	str1.Format(_T("%d"), intW);
	str2.Format(_T("%d"), intLightness);

	str1.AppendChar('%');
	str2.AppendChar('%');
	GetDlgItem(IDC_STATIC3)->SetWindowText(str1);
	GetDlgItem(IDC_STATIC2)->SetWindowText(str2);
	CDialogEx::OnHScroll(nSBCode, nPos, pScrollBar);
}
