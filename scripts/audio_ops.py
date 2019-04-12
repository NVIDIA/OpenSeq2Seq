import librosa

class AudioFile(object):
  def __init__(self, filepath=None, sr=16000, shift=-0.15, idx_to_time=0.02):
    self.sr = sr
    self.shift = shift
    self.idx_to_time = idx_to_time
    if filepath is not None:
      self.audio, _ = librosa.load(filepath, sr=self.sr)
      self.time = len(self.audio)
  def split_into_sentences(self, sentences, out=True, out_file_path="", custom_name=""):
    curr_time = 0
    if self.audio is None:
      raise ValueError("No audio file to split")
    else:
      i = 1
      if out:
        csv_rows = []
      for sentence in sentences:
        out_name = "{}_sentence_{}".format(custom_name, i)
        transcript = sentence[0]
        start, end = sentence[1]*self.idx_to_time+self.shift, sentence[2]*self.idx_to_time+self.shift
        sound = self.audio[int(start * self.sr):int(end * self.sr)]
        curr_time += len(sound)
        out_file = out_file_path + out_name
        librosa.output.write_wav(out_file, sound, self.sr)
        if out:
          csv_rows.append([out_file, transcript])
        i += 1
    print("Parsing and alignment ratio = {}".format(curr_time/self.time))
    if out:
      return csv_rows