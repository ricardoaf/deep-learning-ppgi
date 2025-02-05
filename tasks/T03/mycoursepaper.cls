% coursepaper.cls
%------------------------------------------------------------------------%
%% This is a class (.cls) file for LaTeX2e.
%% 
%% Author:      Norman R. Gall
%%              Department of Philosophy
%%              The University of Calgary
%%              Calgary, Alberta
%%
%%              ngall@ucalgary.ca
%%
%% Feel free to copy, modify, and distribute.
%% I am interested in all changes you make.
%% Send changes to ngall@ucalgary.ca
%%
%% This file is based on the default article.cls file 
%%
%% General Notes:
%%
%% I have taken quite a few liberties in the template to cohere with 
%% idiosyncratic preferences in reading student's papers. I think that
%% while other instructors might be more liberal, there will be few who 
%% are more conservative in their requirements. 
%%
%% My purpose in writing this .cls is to provide students with a single 
%% straightforward template from which to generate clear, clean, and 
%% simple papers that are easy to read and identify by instructors.
%%
%%
%% I have assumed onecolumn and onepage. In my experience, no one likes to 
%% see columned or duplex papers.
%%
% Version History
% ---------------
% v0.1  2002-02-16	Test of Class file                       
% v1.0  2002-02-21	Cleaned up comments and added \coursesection
% v2.0  2003-08-19	Major revision to make template generic
%					
% - Command Notes ------------------------------
%
% new commands:	\studentnumber, \college, \coursename, \coursenumber, 
% 			    \coursesection, \instructor
%
% All of these commands go in the preamble and are pretty 
% straightforward. You use them just like the \author
% command. I have smart-assed defaults programmed, so you'd better 
% remember to insert your own values.
%
% I have included these as these are what I want to see on each and every 
% paper I receive. If there are other items that you might find helpful, 
% e-mail the suggestion.
%
% Page Styles
%
%	\pagestyle{headings} and pagestyle{plain} work as per article.cls 
%    except that 
%			a) when using titlepage, there is no page number printed on 
%               the title page and the title page is not counted.
%			b) when not using titlepage, there is no page number printed 
%               on the first page but it is counted.
%			c) when using titlepage and there is an abstract, there is no 
%               page number printed on the title page or the abstract page 
%               and neither the title page nor abstract page is counted.
%			d) when not using titlepage and there is an abstract, there 
%               is no page number printed on the first page but it is 
%               counted. The abstract is inserted immediately after the 
%               title matter.
%
%	\pagestyle{myheadings) works as above except it has the marks shifted 
%    to the right for clarity so that students can place their name on 
%    each page, if required
%
% Abstract
% 
% Some disciplines regularly use the abstract in papers, so I have dealt 
% with it. If you use the titlepage option and included the abstract right 
% after the title material, the abstract will be placed on a second page 
% by itself after the titlepage. The main body will then begin on the next
% page, with page numbering beginning there.
%
% Draft and Final Modes
% 
% The draft and final options actually do something other than show you 
% line overruns. The draft option single spaces everything and gives you 
% slightly smaller margins. The final option gives you wider margins (for
% comments---students never leave enough room for instructor marginalia), 
% and double spaces everything except the abstract, quotations, quotes, 
% verses, and footnotes. None of these things should be double-spaced in a 
% final paper, in my view. The bibliography (`References' has been 
% renamed) will be double spaced. You can use a \spacing{1} before your 
% \bibliography command to change that behaviour.
%
% There is also an \epigraph command at the end, if you are into that sort 
% of thing in your papers.
%
% - Identification -------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mycoursepaper}[2019/04/01 v0.9]
% Initial Code ---------------------------------------
%
\RequirePackage{ifthen}
%
% - Declaration of Options -----------------------------
%
% saving taken from dissertation.sty with modification
%
\newboolean{saving}
%
\DeclareOption{draft}{\setboolean{saving}{true}}
\DeclareOption{final}{\setboolean{saving}{false}}

\PassOptionsToClass{letterpaper,oneside,onecolumn}{article}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{draft}
%
% - Execution of Options -------------------------------
%
\ProcessOptions
%
% - Package Loading ------------------------------------
%
\LoadClass{article}

%
% - Main Code ------------------------------------------
%
%     some redefinitions I need later
%
\let\oldquote\quote
\let\oldendquote\endquote
\let\oldquotation\quotation
\let\oldendquotation\endquotation
\let\oldverse\verse
\let\oldendverse\oldverse
\let\oldbibliography\bibliography
\let\oldendbibliography\endbibliography
\let\oldabstract\abstract
\let\oldendabstract\endabstract
%
%     some definitions and defaults that are convenient

%
\def\college#1{\gdef\@college{#1}}
\def\coursename#1{\gdef\@coursename{#1}}
\def\coursenumber#1{\gdef\@coursenumber{#1}}
\def\coursesection#1{\gdef\@coursesection{#1}}
\def\studentnumber#1{\gdef\@studentnumber{#1}}
\def\instructor#1{\gdef\@instructor{#1}}
%
\gdef\@author{I.M.A. Student}
\gdef\@title{The Socio-Economic Layering of Sub-Atomic Particle Cultures}
\gdef\@date{\today}
\gdef\@college{School of Hard Knocks}
\gdef\@coursename{Sociology of Physics}
\gdef\@coursenumber{SOCI4568}
\gdef\@coursesection{L01}
\gdef\@studentnumber{00000000}
\gdef\@instructor{Professor Y.R.U StillHere}
%
% Spacing and margins
%
% These margins are set for letter size paper for North American tastes.
% Other users will have to make the relevant adjustments here or in their 
% LaTeX source file
%
% \spacing - redefine \baselinestretch and recompute \baselineskip
%
\newcommand{\spacing}[1]{\renewcommand{\baselinestretch}{#1}%
			\large\normalsize}
%
\ifthenelse{\boolean{saving}}
{       % settings for draft
		\setlength {\topmargin}{0.0in}	 % top margin    1.0 inch
        \setlength {\textheight}{8.5in}	 % bottom margin 1.0 inch
        \setlength {\oddsidemargin}{0.0in}	 % left margin   1.0 inch
		\setlength {\evensidemargin}{0.0in}	 % right margin  1.0 inch
        \setlength {\textwidth}{6.5in}	 % right margin  1.0 inch
        \spacing{1}						 % single spacing for draft
}
{       % settings for final
		\setlength {\topmargin}{0.0in}	 % top margin    1.0 inch
        \setlength {\textheight}{8.5in}	 % bottom margin 1.0 inch
        \setlength {\oddsidemargin}{0.5in} % left margin    1.5 inch
		\setlength {\evensidemargin}{0.5in}	 % right margin  1.5 inch
        \setlength {\textwidth}{5.5in}	 % right margin  1.5 inch
        \spacing{2}						 % double spacing for final
        \renewenvironment{quote}			 % redef as single spaced
			{\oldquote\spacing{1}}
			{\oldendquote\spacing{2}}
		\renewenvironment{quotation}		 % redef as single spaced
			{\oldquotation\spacing{1}}
			{\oldendquotation\spacing{2}} 
		\renewenvironment{verse}		     % redef as single spaced
			{\oldquotation\spacing{1}}
			{\oldendquotation\spacing{2}} 
}
\setlength {\footnotesep}{14pt}			  % baseline skip for fn's 1st line
\setlength {\headheight}{0.2in}			  % make room for header
\setlength {\headsep}{0.2in}				  % modest header separation
\setlength {\parskip}{0.2in}				  % set a paragraph skip
\setlength {\parindent}{0.2in}			  % I hate unindented first lines
\renewenvironment{bibliography}[1]		  % automatically enter toc line
	{\addcontentsline{toc}{chapter}{Bibliography}\oldbibliography {#1}}
	{\oldendbibliography}
%
% Make footnotes single spaced
%
%			code shamelessly stolen from setspace.sty 
%				written by Geoffrey Tobin <G.Tobin@latrobe.edu.au>
%
\long\def\@footnotetext#1{%
  \insert\footins{%
    \def\baselinestretch {1}%
    \reset@font\footnotesize
    \interlinepenalty\interfootnotelinepenalty
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth
    \@parboxrestore
    \protected@edef\@currentlabel{%
      \csname p@footnote\endcsname\@thefnmark
    }%
    \color@begingroup
      \@makefntext{%
        \rule\z@\footnotesep\ignorespaces#1\@finalstrut\strutbox}%
    \color@endgroup}}
%
% redefine the titlematter
%
\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \thispagestyle{empty}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  %
  \global\let\@studentnumber\@empty
  \global\let\@coursenumber\@empty
  \global\let\@coursename\@empty
  \global\let\@coursesection\@empty
  \global\let\@instructor\@empty
  \global\let\@college\@empty
  %
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
%
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{flushleft}%
  	\begin{spacing}{1}%
    {\sffamily \LARGE \@title \par}%
    \sffamily \@date
    \vskip 1em%
    {\sffamily \large\lineskip .75em\@author}\\%
    \sffamily \@studentnumber%
    \vskip 1em%
    \sffamily \@college\\%
    \vskip 1em%
    \sffamily \@coursenumber\hskip 6pt\@coursesection\hskip 6pt%
			\@coursename\\%
    \sffamily Para: \@instructor%
    \par%
    \end{spacing}%
  \end{flushleft}\hrule\vskip 1em\par
  \par
  \vskip 1.5em}
%
% make section titles less obnoxious
%
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}
%
% Set up headers
%
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage -- \slshape\leftmark\hfil}%
    \def\@oddhead{\hfil{\slshape\rightmark} -- \thepage}%
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
    }
%
% Position the abstract depending on presence of titlepage
%
\if@titlepage
  \renewenvironment{abstract}{%
	  \titlepage
      \null\vfil
      \@beginparpenalty\@lowpenalty
      \begin{center}%
        \bfseries \abstractname
        \@endparpenalty\@M
      \end{center}}%
     {\par\vfil\null\endtitlepage}
\fi
%
% change some names
%
\renewcommand\contentsname{Table of Contents}
\renewcommand\refname{Bibliography}
%
%	\epigraph{text}{byline}
%
\newcommand{\epigraph}[2]{\begin{flushright}\begin{minipage}{4in}%
			\spacing{1}#1\begin{flushright}#2\end{flushright}%
			\hrule\end{minipage}\vskip 12pt\end{flushright}}
%
%% End of file `coursepaper.cls'.
