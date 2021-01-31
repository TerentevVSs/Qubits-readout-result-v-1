import numpy as np
from qiskit import IBMQ
from qiskit import pulse  # This is where we access all of our Pulse features!
from qiskit.pulse import Play
from qiskit.pulse import pulse_lib  # This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit import assemble
from qiskit.tools.monitor import job_monitor


"""
Программа для сбора данных с квантовго компьютера IBM Armonk
"""


IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"
dt = backend_config.dt
print(f"Sampling time: {dt * 1e9} ns")
backend_defaults = backend.defaults()

qubit = 0

# достаем из текстового файла частоту кубита и амплитуду pi импулса
with open("qubit_params.txt", "r") as f:
    rough_qubit_frequency, pi_amp = list(map(lambda x: float(x.strip()), f.readlines()))

us = 1.0e-6  # Microseconds
scale_factor = 1e-14

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

drive_sigma_us = 0.075  # This determines the actual width of the gaussian
drive_samples_us = drive_sigma_us * 8  # This is a truncating parameter, because gaussians don't have

drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us / dt)  # The width of the gaussian in units of dt
drive_samples = get_closest_multiple_of_16(drive_samples_us * us / dt)  # The truncating parameter in units of dt
drive_amp = 0.3
# Drive pulse samples
drive_pulse = pulse_lib.gaussian(duration=drive_samples,
                                 sigma=drive_sigma,
                                 amp=drive_amp,
                                 name='freq_sweep_excitation_pulse')

# берем амплитуду из второй строки текстового файла
pi_pulse = pulse_lib.gaussian(duration=drive_samples,
                              amp=pi_amp,
                              sigma=drive_sigma,
                              name='pi_pulse')

# Find out which group of qubits need to be acquired with this qubit
meas_map_idx = None
for i, measure_group in enumerate(backend_config.meas_map):
    if qubit in measure_group:
        meas_map_idx = i
        break
assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"

inst_sched_map = backend_defaults.instruction_schedule_map
measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])

### Collect the necessary channels
drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)


# Create two schedules

# Ground state schedule
gnd_schedule = pulse.Schedule(name="ground state")
gnd_schedule += measure

# Excited state schedule
exc_schedule = pulse.Schedule(name="excited state")
exc_schedule += Play(pi_pulse, drive_chan)  # We found this in Part 2A above
exc_schedule += measure << exc_schedule.duration

# Execution settings
num_shots = 1024*8

gnd_exc_program = assemble([gnd_schedule, exc_schedule],
                           backend=backend,
                           meas_level=0,
                           meas_return='single',
                           shots=num_shots,
                           schedule_los=[{drive_chan: rough_qubit_frequency}] * 2)

# print(job.job_id())

def loop():
    job = backend.run(gnd_exc_program)
    job_monitor(job)

    gnd_exc_results = job.result(timeout=120)

    gnd_results = gnd_exc_results.get_memory(0)[:, qubit]*scale_factor
    exc_results = gnd_exc_results.get_memory(1)[:, qubit]*scale_factor


    # записываем данные в .txt файл
    with open("0state_08_12_2020.txt", mode="a") as f:
        #f.writelines("Состояние |0>")
        f.writelines(list(map(lambda x: str(x) + "\n", gnd_results)))
        #f.writelines("Состояние |1>")
        #f.writelines(list(map(lambda x: str(x) + "\n", exc_results)))

    with open("1state_08_2020.txt", mode="a") as f:
        #f.writelines("Состояние |0>")
        f.writelines(list(map(lambda x: str(x) + "\n", exc_results)))
        #f.writelines("Состояние |1>")
        #f.writelines(list(map(lambda x: str(x) + "\n", exc_results)))


for i in range(5):
    loop()