rm -rf dist
rm -rf NpDataAnalysis_v$1_mac
mv __init__.py init_archive

pyinstaller -w --noconfirm NpDataAnalysis_v0.2.py &&\
	mv dist/NpDataAnalysis_v0.2.app dist/NpDataAnalysis_v$1.app && \
	cp -R dist NpDataAnalysis_v$1_mac 
#	zip -r NpDataAnalysis_v$1_mac.zip  NpDataAnalysis_v$1_mac

mv init_archive __init__.py
