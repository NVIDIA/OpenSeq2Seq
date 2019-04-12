import os
import librosa
import soundfile as sf
import csv
import re
from . import text_cleaner
from difflib import SequenceMatcher as SM
extensions = ("wav", "flac", "mp3")

def speech_to_wav(inp_file,outdirname, freq=16000):

  if not os.path.isdir(outdirname):
    os.makedirs(outdirname)

  filename = inp_file.split("/")[-1].split(".")[0]
  wave, sr = librosa.load(inp_file, sr=freq)
  outname = os.path.join(outdirname, filename + ".wav")
  sf.write(outname, wave, sr)
  return outname


def dir_to_wav(dirname, freq=16000, csv_out=False, csv_fname="sample.csv"):
  files = os.listdir(dirname)
  output_rows= []

  for f in files:
    ext = f.split(".")[-1]
    if ext in extensions:
      wav_filename = speech_to_wav(f,os.path.dirname(f)+"/wav",freq)
      output_rows.append(wav_filename)

  if csv_out:
    with open(os.path.dirname(f) + csv_fname, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
      for row in output_rows:
        writer.writerow([row])
  else:
    return output_rows


def get_chapters(book):
  import re
  chapters = re.compile("CHAPTER .*\n\n\n").split(book)
  return chapters[1:]

def process_chapters(chapter):
  rows = []
  lines = re.compile('[.?]').split(chapter)
  for line in lines:
    new_line = text_cleaner.english_cleaners(line)
    rows.append(new_line)
  return rows

def get_first_match_word(match, row):
  start = match.b
  if start!=0 and row[start-1]!=" ":
    while(row[start]!=" "):
      start+=1
  return start

def get_last_match_word(match, row):
  end = match.b+match.end
  if end!=len(row)-1 and row[end]!=" ":
    while(row[end]!=" "):
      end-=1
  return end

def detect_good_point(actual, prediction, match, place=0):
  row_words = actual.split(" ")
  true_word = row_words[place]
  start = match.a
  end = match.a + match.size
  few_predictions = prediction[start:end]
  predicted_word = few_predictions.split(" ")[place]
  if predicted_word == true_word:
    return True
  return False


def find_start_match(matches, prev):
  for i, match in enumerate(matches):
    if match.size > 10 and match.a>=prev:
      return i, match
  return -1, None


def find_end_match(matches, idx):
  for i in range(len(matches) - 1, idx, -1):
    if matches[i].size > 10:
      return matches[i]
  return None

def get_full_matches(model_output, actual_outputs):
  prev_idx = -1
  next_start = -1
  prev_bad_end = False
  prev = -1
  res = []
  for row in actual_outputs:
    idx = 0
    row = row.strip()
    matches = SM(None, model_output, row).get_matching_blocks()
    if len(matches) > 1:
      if next_start != -1:
        start = next_start
      else:
        idx, match = find_start_match(matches, prev)
        if idx == -1:
          if len(res) > 0 and res[-1][-1] == -1:
            prev_bad_end = False
            res.pop()
          continue
        good_start = detect_good_point(row, model_output, match, 0)
        if good_start:
          start = match.a
          if prev_bad_end and len(res)>0:
            res[-1][-1] = start - 1
        else:
          if len(res) > 0 and res[-1][-1] == -1:
            prev_bad_end = False
            res.pop()
          continue
      match = find_end_match(matches, idx)
      if match is None:
        next_start = -1
        res.append([row, start, next_start])
        prev_bad_end = True
        continue
      good_end = detect_good_point(row, model_output, match, -1)
      if good_end:
        next_start = match.a + match.size + 1
        res.append([row, start, next_start])
        prev_bad_end = False
      else:
        next_start = -1
        res.append([row, start, next_start])
        prev_bad_end = True
      prev = start
    else:
      next_start = -1
      if len(res) > 0 and res[-1][-1] == -1:
        # print("Popping {}".format(res[-1]))
        prev_bad_end = False
        res.pop()
      continue
  if len(res) > 0 and res[-1][-1] == -1:
    res.pop()
  return res

def get_first_match_word(match, row):
  start = match.b
  if start > 0 and start < len(row) and row[start - 1] != " ":
    while (row[start] != " "):
      start += 1
      if start == len(row):
        return start
  return start


def get_last_match_word(match, row):
  end = match.b + match.size
  if end < len(row) and row[end] != " ":
    while (row[end] != " "):
      end -= 1
  return end

def get_partial_matches(model_output, actual_outputs, results):
  prev_idx = -1
  cnt = 0

  for row in actual_outputs:
    row = row.strip()
    matches = SM(None, model_output, row).get_matching_blocks()
    first_big_match = None
    last_big_match = None

    if len(matches) > 1:
      flag = 1
      for match in matches:
        if match.size > 11:
          if flag == 1:
            first_big_match = match
            last_big_match = match
            flag = 0
          else:
            last_big_match = match

      if first_big_match is not None and last_big_match is not None:
        start = get_first_match_word(first_big_match, row)
        end = get_last_match_word(last_big_match, row)

        if start < end and end - start < 2 * len(row):
          diff_start = start - first_big_match.b
          diff_end = end - (last_big_match.b + last_big_match.size)
          predicted_start = first_big_match.a + diff_start
          predicted_end = last_big_match.a + last_big_match.size + diff_end

          if prev_idx <= predicted_start and predicted_end-predicted_start<2*len(row):
            if row not in results:
              sentence = row[start:end].strip()

              if len(sentence.split(" ")) > 2: #only sentences with more than two words
                results[sentence] = {"start": predicted_start, "end": predicted_end}
            prev_idx = predicted_end
            cnt += 1
  return results

def get_best_match(model_output, actual_outputs):
  results = {}
  res = get_full_matches(model_output, actual_outputs)
  for r in res:
    key = r[0]
    results[key] = {"start": r[1], "end": r[2]}
  results = get_partial_matches(model_output, actual_outputs, results)
  matches = []
  for r in results:
    if results[r]["end"] - results[r]["start"] <= 2 * len(r) and results[r]["end"] > results[r]["start"]:
      start, end = results[r]["start"], results[r]["end"]
      matches.append([r,model_output[start:end], start, end])
  matches = sorted(matches, key=lambda s:s[2])
  return matches

def find_word_idx(text, actual, start, end):
  sub_text = text[start:end].strip()
  words = sub_text.split(" ")
  word_num = len(text[:start].strip().split(" "))
  return [actual.strip(), word_num, word_num + len(words)]

def char_to_word(alignments, predicted):
  word_alignments = []
  for alignment in alignments:
    word_alignments.append(find_word_idx(predicted, alignment[0], alignment[2], alignment[3]))
  return word_alignments

def sentence_alignment(word_alignment, start_word_times):
  sentences = []
  for alignment in word_alignment:
    start, end = alignment[1], alignment[2]
    sentences.append([alignment[0], float(start_word_times[start]), float(start_word_times[end])])
  return sentences

def clean_book(book_file):
  book_text = open(book_file,'r').read()
  chapters = get_chapters(book_text)
  cleaned_chapters = []
  for chapter in chapters[:-1]:
    lines  = process_chapters(chapter)
    cleaned_chapters.append(lines)
  last_chapter = re.compile("THE END").split(chapters[-1])[0]
  lines = process_chapters(last_chapter)
  cleaned_chapters.append(lines)
  return cleaned_chapters


