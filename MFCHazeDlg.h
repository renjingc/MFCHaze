
// MFCHazeDlg.h : 头文件
//
#pragma once
#include "guidedfilter.h"
#include "deHaze.h"
#include "CvvImage.h"
#include "afxcmn.h"

#include <strstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include "afxwin.h"


using namespace std;
using namespace cv;

// CMFCHazeDlg 对话框
class CMFCHazeDlg : public CDialogEx
{
// 构造
public:
	CMFCHazeDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_MFCHAZE_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持
	int LoadPicture(string filePath);
	int showImage(Mat image, UINT ID);
	int saveImage(Mat image, string fileName);
	int saveVideo(string fileName);

	Mat image;
	Mat resultImage;
	VideoCapture camera;
	COLORREF BKcolor;

// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpenimage();
	afx_msg void OnBnClickedOpenvideo();
	afx_msg void OnBnClickedStarthazeimage();
	afx_msg void OnBnClickedStarthazevideo();
	afx_msg void OnBnClickedSaveimage();

	CSliderCtrl m_ctrlSlider1;
	CSliderCtrl m_ctrlSlider2;
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	CComboBox m_algorithm;
	CStatic m_time;
	CStatic m_snr;
};
