\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{Thesis}
              [2007/22/02 v1.0
   LaTeX document class]
\def\baseclass{book}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\def\@checkoptions#1#2{
  \edef\@curroptions{\@ptionlist{\@currname.\@currext}}
  \@tempswafalse
  \@tfor\@this:=#2\do{
    \@expandtwoargs\in@{,\@this,}{,\@curroptions,}
    \ifin@ \@tempswatrue \@break@tfor \fi}
  \let\@this\@empty
  \if@tempswa \else \PassOptionsToClass{#1}{\baseclass}\fi
}
\@checkoptions{11pt}{{10pt}{11pt}{12pt}}
\PassOptionsToClass{a4paper}{\baseclass}
\ProcessOptions\relax
\LoadClass{\baseclass}
\newcommand
\bhrule{\typeout{------------------------------------------------------------------------------}}
\newcommand
\Declaration[1]{
	\btypeout{Declaration of Authorship}
	\addtotoc{Declaration of Authorship}
	\thispagestyle{plain}
	\null\vfil
	\begin{center}{
    	\huge\bf Declaration of Authorship\par
    }
    \end{center}
{\normalsize #1}
\vfil\vfil\null
}
\newcommand
\btypeout[1]{\bhrule\typeout{\space #1}\bhrule}
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{2.0ex plus0.5ex minus0.2ex}
\usepackage{vmargin}
\RequirePackage[utf8]{inputenc}
\setmarginsrb { 1in}  % left margin
              { 0.6in}  % top margin
              { 1.0in}  % right margin
              { 0.8in}  % bottom margin
              {  20pt}  % head height
              {0.25in}  % head sep
              {   9pt}  % foot height
              { 0.3in}  % foot sep
%-----------------------------------------------------------------------
\raggedbottom
\setlength{\topskip}{1\topskip \@plus 5\p@}
\doublehyphendemerits=10000       % No consecutive line hyphens.
\brokenpenalty=10000              % No broken words across columns/pages.
\widowpenalty=9999                % Almost no widows at bottom of page.
\clubpenalty=9999                 % Almost no orphans at top of page.
\interfootnotelinepenalty=9999    % Almost never break footnotes.
\usepackage{fancyhdr}
\lhead[ \rm \thepage]{ \fancyplain{}{\sl{\rightmark}}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{
	\btypeout{\thechapter\space #1}
	\markboth{\@chapapp\ \thechapter\#1}{\@chapapp\ \thechapter\ #1}
}
\renewcommand{\sectionmark}[1]{}
\renewcommand{\subsectionmark}[1]{}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\thispagestyle{empty}
\if@twocolumn\hbox{}\newpage\fi\fi\fi}
\usepackage{amsmath,amsfonts,amssymb,amscd,amsthm,xspace}
\theoremstyle{plain}
\newtheorem{teorema}{Teorema}[section]
\newtheorem{lema}[teorema]{Lema}
\newtheorem{ejemplo}[teorema]{Ejemplo}
\newtheorem{definicion}[teorema]{Definici\'{o}n}
\newtheorem{proposicion}[teorema]{Proposici\'{o}n}
\newtheorem{resultado}[teorema]{Resultado}
\usepackage[centerlast,small,sc]{caption}
\setlength{\captionmargin}{20pt}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\dbltopfraction}{.85}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{.75}
\usepackage{graphicx}
\usepackage{epstopdf}
\usepackage{booktabs}
\usepackage{rotating}
\usepackage{listings}
\usepackage{lstpatch}
\lstset{captionpos=b,
        frame=tb,
        basicstyle=\scriptsize\ttfamily,
        showstringspaces=false,
        keepspaces=true}
\lstdefinestyle{matlab} {
        language=Matlab,
        keywordstyle=\color{blue},
        commentstyle=\color[rgb]{0.13,0.55,0.13}\em,
        stringstyle=\color[rgb]{0.7,0,0} }
\usepackage[%
	pdfpagemode=UseOutlines, 
    bookmarks=true,
    bookmarksopen=true,
   	bookmarksopenlevel=0,
    bookmarksnumbered=true,
    hypertexnames=false,
   	colorlinks,
    linkcolor=blue,
    citecolor=blue,
    pagebackref
]{hyperref}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
\pdfstringdefDisableCommands{
   \let\\\space
}
\newcommand*{\director}[1]{\def\director{#1}}
\newcommand*{\titulo}[1]{\def\titulo{#1}}
\newcommand*{\sinodal}[1]{\def\sinodal{#1}}
\newcommand*{\grado}[1]{\def\grado{#1}}
\newcommand*{\autor}[1]{\def\autor{#1}}
\newcommand*{\universidad}[1]{\def\universidad{#1}}
\newcommand*{\departmento}[1]{\def\departamento{#1}}
\newcommand*{\facultad}[1]{\def\facultad{#1}}
\def\primsinodal#1{\def\sinoda{\Large #1}}%
\def\sinoda{}%
\def\segsinodal#1{\def\sinodb{\Large #1}}%
\def\sinodb{}%
\def\tersinodal#1{\def\sinodc{\Large #1}}%
\def\sinodc{}%
\def\ctosinodal#1{\def\sinodd{\Large #1}}%
\def\sinodd{}%
\primsinodal{
		Dr. Fernando Verduzco González
		\\
		Universidad de Sonora, Hermosillo, M\'exico
 }
 \segsinodal{
		Dr. Francisco Armando Carrillo Navarro
		 \\
		 Universidad de Sonora, Hermosillo, M\'exico
 	}
  \tersinodal{
		Dr. Daniel Olmos Liceaga\\
		Universidad de Sonora, Hermosillo, M\'exico
 }
  \ctosinodal{
		Dr. Sa\'ul Díaz Infante Velasco\\
		CONACYT-Universidad de Sonora, Hermosillo, M\'exico
 }
\titulo{
	Modelos Epidemeológicos para la Tuberculosis:
	Resistencia Múltiple, Reinfección Exógena y Bifurcación
	Hacia Atrás
} 
%-------------------------------------------------  
\director{Dr. Saúl Díaz Infante Velasco} 
%-------------------------------------------------   
%-------------------------------------------------   
\grado{Maestr\'ia  en Matem\'aticas} 
%-------------------------------------------------   
\autor{Nohemy Palafox Lacarra} 
%-------------------------------------------------   
\universidad{{Universidad de Sonora}}                 
%-------------------------------------------------              
\departmento{{Departamento de Matem\'aticas}} % 
%-------------------------------------------------
\facultad{{Divisi\'on de Ciencias Exactas y Naturales}}
%-------------------------------------------------                    
\renewcommand\maketitle{
  \btypeout{Title Page}
  \hypersetup{pdftitle={\@title}}
  \hypersetup{pdfsubject=\subjectname}
  \hypersetup{pdfauthor=\authornames}
  \hypersetup{pdfkeywords=\keywordnames}
  \thispagestyle{empty}
  \begin{titlepage}
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \setcounter{footnote}{0}
    \null\vfil
    \vskip 60\p@
    \begin{center}
      \setlength{\parskip}{0pt}
      {\large\textbf{\UNIVNAME}\par}
      \vfill
      {\huge \bf \@title \par}
      \vfill
      {\LARGE by \par}
      \smallskip
      {\LARGE \authornames \par}
      \vfill
      {\large \deptname \par} 
      \bigskip
      \bigskip
      \bigskip
      {\Large \@date \par}
      \bigskip
    \end{center}
    \par
    \@thanks
    \vfil\null
  \end{titlepage}
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
%-------------------------------------------------------------------------------%	
% ABSTRACT PAGE DESIGN
%-------------------------------------------------------------------------------
\newenvironment{abstract}
{
  \btypeout{Abstract Page}
  \thispagestyle{empty}
  \null\vfil
  \begin{center}
    \setlength{\parskip}{0pt}
    {\normalsize \UNIVNAME \par} % University name in capitals
    \bigskip
    {\huge{\textit{Abstract}} \par}
    \bigskip
    {\normalsize \facname \par} % Faculty name
    {\normalsize \deptname \par} % Department name
    \bigskip
    {\normalsize \degreename\par} % Degree name
    \bigskip
    {\normalsize\bf \@title \par} % Thesis title
    \medskip
    {\normalsize by \authornames \par} % Author name
    \bigskip
  \end{center}
}
%-------------------------------------------------------------------------------
{
  \vfil\vfil\vfil\null
  \cleardoublepage
}
\addtocounter{secnumdepth}{1}
\setcounter{tocdepth}{6}
\newcounter{dummy}
\newcommand\addtotoc[1]{
\refstepcounter{dummy}
\addcontentsline{toc}{chapter}{#1}}
\renewcommand\tableofcontents{
\btypeout{Contents}
\addtotoc{Contents}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}
    \@starttoc{toc}
    \if@restonecol\twocolumn\fi
   \cleardoublepage
}\end{spacing}
}
\renewcommand\listoffigures{
\btypeout{List of Figures}
\addtotoc{List of Figures}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}
              {\MakeUppercase\listfigurename}}
    \@starttoc{lof}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\renewcommand\listoftables{
\btypeout{List of Tables}
\addtotoc{List of Tables}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}
    \@starttoc{lot}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\newcommand\listsymbolname{Abbreviations}
\usepackage{longtable}
\newcommand\listofsymbols[2]{
\btypeout{\listsymbolname}
\addtotoc{\listsymbolname}
    \chapter*{\listsymbolname
      \@mkboth{
          \MakeUppercase\listsymbolname}{\MakeUppercase\listsymbolname}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}
\newcommand\listconstants{Physical Constants}
\usepackage{longtable}
\newcommand\listofconstants[2]{
\btypeout{\listconstants}
\addtotoc{\listconstants}
    \chapter*{\listconstants
      \@mkboth{
          \MakeUppercase\listconstants}{\MakeUppercase\listconstants}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}
\newcommand\listnomenclature{Symbols}
\usepackage{longtable}
\newcommand\listofnomenclature[2]{
\btypeout{\listnomenclature}
\addtotoc{\listnomenclature}
    \chapter*{\listnomenclature
      \@mkboth{
          \MakeUppercase\listnomenclature}{\MakeUppercase\listnomenclature}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}
\newcommand\acknowledgements[1]{
\btypeout{Acknowledgements}
\addtotoc{Acknowledgements}
\thispagestyle{plain}
\begin{center}{\huge{\textit{Agradecimientos}} \par}\end{center}
{\normalsize #1}
	\vfil\vfil\null
}
\newcommand\dedicatory[1]{
\btypeout{Dedicatory}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large \sl #1}\end{center}
\vfil\null
\cleardoublepage
}
\renewcommand\backmatter{
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \addtotoc{\bibname}
  \btypeout{\bibname}
  \@mainmatterfalse}
\endinput
