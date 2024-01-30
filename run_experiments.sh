#!/bin/sh

python script/representation.py x264 kbs -m original --data-dir data --output results
python script/representation.py x264 kbs -m tsne -d 3 --data-dir data --output results
python script/representation.py x264 kbs -m pca -d 3 --data-dir data --output results
python script/representation.py x264 kbs -m embed -d 3 --data-dir data --output results
python script/representation.py x264 kbs -m embed -d 32 --data-dir data --output results

python script/representation.py gcc size -m original --data-dir data --output results
python script/representation.py gcc size -m tsne -d 3 --data-dir data --output results
python script/representation.py gcc size -m pca -d 3 --data-dir data --output results
python script/representation.py gcc size -m embed -d 3 --data-dir data --output results
python script/representation.py gcc size -m embed -d 32 --data-dir data --output results

python script/representation.py imagemagick size -m original --data-dir data --output results
python script/representation.py imagemagick size -m tsne -d 3 --data-dir data --output results
python script/representation.py imagemagick size -m pca -d 3 --data-dir data --output results
python script/representation.py imagemagick size -m embed -d 3 --data-dir data --output results
python script/representation.py imagemagick size -m embed -d 32 --data-dir data --output results

python script/representation.py poppler size -m original --data-dir data --output results
python script/representation.py poppler size -m tsne -d 3 --data-dir data --output results
python script/representation.py poppler size -m pca -d 3 --data-dir data --output results
python script/representation.py poppler size -m embed -d 3 --data-dir data --output results
python script/representation.py poppler size -m embed -d 32 --data-dir data --output results

