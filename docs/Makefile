all:
	mkdir -p ./build
	rm -rf ./build/sphinx
	bash get_example_sources.sh
	cd doxygen && make
	python3 generate_rst_pages_doxygen_groups.py ./build/doxygen/xml ./sphinx/source/guide-user
	cd sphinx && make html
	mv -f ./build/doxygen ./build/sphinx/html/
