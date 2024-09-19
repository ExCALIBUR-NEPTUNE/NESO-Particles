# Trims the first and last line from the sample sources as these lines disable
# and enable the formatting.

EXAMPLE_SOURCES="../test/example_sources/"
EXAMPLE_OUTPUTS="sphinx/source/example_sources"

# workaround macOS utils and assume "brew install coreutils" has been ran
HEAD=head
if hash ghead 2>/dev/null; then
    HEAD=ghead
fi

mkdir -p $EXAMPLE_OUTPUTS

for fx in $EXAMPLE_SOURCES/*.hpp; do
echo $fx
echo $(basename -- $fx)
echo $EXAMPLE_OUTPUTS/$(basename -- $fx)
cat $fx | tail -n +2 | $HEAD -n-1 > $EXAMPLE_OUTPUTS/$(basename -- $fx)
done
