# Trims the first and last line from the sample sources as these lines disable
# and enable the formatting.

EXAMPLE_SOURCES="../test/example_sources/"
EXAMPLE_OUTPUTS="sphinx/source/example_sources"
mkdir -p $EXAMPLE_OUTPUTS

for fx in $EXAMPLE_SOURCES/*.hpp; do
echo $fx
echo $(basename -- $fx)
echo $EXAMPLE_OUTPUTS/$(basename -- $fx)
cat $fx | tail -n +2 | head -n-1 > $EXAMPLE_OUTPUTS/$(basename -- $fx)
done
