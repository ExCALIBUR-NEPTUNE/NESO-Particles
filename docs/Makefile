all:
	mkdir -p ./build
	cd doxygen && make
	cd sphinx && make html
	mv ./build/doxygen ./build/sphinx/html/
