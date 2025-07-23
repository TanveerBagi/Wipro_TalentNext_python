def find_meeting_details(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)
    if num_lines == 0:
        return "File is empty."

    if num_lines > 12:
        hour = num_lines - 12
        period = "PM"
    else:
        hour = num_lines
        period = "AM"

    meeting_time = f"{hour} {period}"

    words = []
    for line in lines:
        words.extend(line.strip().split())

    word_count = {}
    for word in words:
        word = word.strip('.,!?').capitalize()
        word_count[word] = word_count.get(word, 0) + 1

    meeting_place_word = max(word_count, key=word_count.get)
    meeting_place = f"{meeting_place_word} Street"

    return f"Meeting Time: {meeting_time}\nMeeting Place: {meeting_place}"



file_path = "message.txt"  #Replace message.txt with the text filw
print(find_meeting_details(file_path))
