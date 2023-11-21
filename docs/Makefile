all:
	mkdir -p ./build
	rm -rf ./build/sphinx
	bash get_example_sources.sh
	cd doxygen && make
	cd sphinx && make html
	mv -f ./build/doxygen ./build/sphinx/html/
