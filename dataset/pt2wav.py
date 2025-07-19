import numpy as np
import torch
import pretty_midi
from midi2audio import FluidSynth
import glob
from tqdm import tqdm
import os

"""
before running this script, you need to install fluidsynth
!apt update
!apt install fluidsynth
!cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./font.sf2
"""

def convert_to_midi(vector, note_split=128):
    """
    convert vector to midi
    
    params
    -------------
    vector: numpy.ndarray #shape=(2, 1024, 128)
    note_split: int #note split
    
    returns
    ---------
    piano: object #pretty_midi object
    """
    vector = ((vector +1)*63.5).clip(0, 127)
    shapes = vector.shape
    instrument_name = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=instrument_name)

    #set parms
    start_time = 0.0
    end_time = 0.0
    tempo = 120
    time_split = 60/(tempo*note_split/4)

    #add notes
    for p in range(shapes[-1]):
        velocity = 0
        start_time = 0.0
        end_time = 0.0
        durting = False #extend note or not
        on = 0 #note on or off
        for t in range(shapes[-2]):
            if not durting:
                on = int(round(vector[1][t][p]))
                velocity = int(round(vector[0][t][p]))
            end_time += time_split
    
            #note on or off in next time
            if on > 0:
                if t != shapes[-2]-1:
                    sutain = int(round(vector[0][t+1][p]))
                    if sutain > 0 and int(round(vector[1][t+1][p])) < 1:
                        durting = True
                        continue
                    else:
                        durting = False
            else:
                start_time = end_time
                durting = False
                continue      
    
            #add note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(p),
                start=start_time, 
                end=end_time
            )
            piano.notes.append(note)
            start_time = end_time
    
    return piano
    
def npy2audio(npy_path, audio_path):
    """
    convert npy to audio

    params
    ----------
    npy_path: str #path to npy file
    audio_path: str #path to audio file
    """
    # load midi
    midi_inst = np.load(npy_path, allow_pickle=True).item()["midi"]
    midi_inst = convert_to_midi(midi_inst)
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(midi_inst)
    # convert audio
    fs = FluidSynth(sound_font='font.sf2')
    midi_path = "tmp.mid"
    midi.write(midi_path)
    fs.midi_to_audio(midi_path, audio_path)

def convert_all_npydata(npy_dir, audio_dir):
    """
    get npy and audio pathes

    params
    ----------
    npy_dir: str #path to npy directory
    """
    npy_pathes = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    for idx, npy_path in tqdm(enumerate(npy_pathes), desc="converting npy to audio"):
        number = str(idx).zfill(len(str(len(npy_pathes)))+1)
        audio_path = os.path.join(audio_dir, f"{number}.wav")
        npy2audio(npy_path, audio_path)

def run():
    pt_path = "/raid/koki-sakurai/model/train/pretrained/sample/finetuning-1600-100/sample_10_B.pt"
    audio_dir = f"data/audio/samples/finetuning-1600-100/sample_10_B"
    os.makedirs(audio_dir, exist_ok=True)
    data = torch.load(pt_path, map_location="cpu")
    data = data["midi"].detach().numpy()
    for idx, d in tqdm(enumerate(data)):
        number = str(idx).zfill(len(str(len(data)))+1)
        audio_path = os.path.join(audio_dir, f"{number}.wav")
        midi_inst = convert_to_midi(d)
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(midi_inst)
        # convert audio
        fs = FluidSynth(sound_font='font.sf2')
        midi_path = "tmp.mid"
        midi.write(midi_path)
        fs.midi_to_audio(midi_path, audio_path)


if __name__ == '__main__':
    run()