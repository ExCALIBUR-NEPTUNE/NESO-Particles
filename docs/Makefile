all:
	mkdir -p ./build
	rm -rf ./build/sphinx
	cd doxygen && make
	cd sphinx && make html
	mv -f ./build/doxygen ./build/sphinx/html/
