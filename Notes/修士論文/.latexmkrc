$latex         = 'uplatex %O -no-guess-input-enc -interaction=nonstopmode -synctex=1 -kanji=utf8 %S';
$bibtex        = 'upbibtex %O %B -kanji=utf8';
$dvipdf        = 'dvipdfmx %O -o %D %S';
$pdf_mode      = 3; # use dvipdfmx
