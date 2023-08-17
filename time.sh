echo "begins..."
timer_start=`date "+%Y-%m-%d %H:%M:%S"`
echo "start time: $timer_start"


timer_end=`date "+%Y-%m-%d %H:%M:%S"`
echo "end time：$timer_end"

start_seconds=$(date --date="$timer_start" +%s);
end_seconds=$(date --date="$timer_end" +%s);
echo "total time: "$((end_seconds-start_seconds))“s”