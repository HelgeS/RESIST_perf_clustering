#!/bin/sh

# python script/representation.py x264 -p kbs -m original --data-dir data --output results
# python script/representation.py x264 -p kbs -m tsne -d 3 --data-dir data --output results
# python script/representation.py x264 -p kbs -m pca -d 3 --data-dir data --output results
# python script/representation.py x264 -p kbs -m embed -d 3 --data-dir data --output results
# python script/representation.py x264 -p kbs -m embed -d 32 --data-dir data --output results

python script/representation.py poppler -p size -m original --data-dir data --output results
python script/representation.py poppler -p size -m tsne -d 3 --data-dir data --output results
python script/representation.py poppler -p size -m pca -d 3 --data-dir data --output results
python script/representation.py poppler -p size -m embed -d 3 --data-dir data --output results
python script/representation.py poppler -p size -m embed -d 32 --data-dir data --output results
