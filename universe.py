import os.path
import math
import random
import datetime

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from gmpy2 import mpz
from mpl_toolkits.mplot3d import Axes3D
from markovnamegen import MarkovGenerator

# universe generation constants (or values which probably wont be tweaked)
max_planet_mass = 2.85e32
min_planet_mass = 1e15
star_freq = .4  # results in about 33% bodies being stars, lowering causes many to not reach threshold
G = 6.67e-11  # gravitational constant
k = 8314  # gas constant
s = 5.67e-8  # Stefan-Boltzmann constant, used in determining surface temp
c = 2.998e+8  # speed of light
AU = 1.496e11  # astronomical unit -> m
solar_luminosity = 3.8328e26
solar_mass = 1.989e30
solar_radius = 6.955e8
comp_variance = 0.5  # a body may have +- this much relative amount of an element
starting_spin = 2  # max spin velocity in rad/s
element_dict = {'H': [73.9, 1, 90],
                'He': [24.0, 4, 180],
                'O': [10.4, 16, 1430],
                'C': [4.6, 12, 2260],
                'Ne': [1.34, 20, 900],
                'Fe': [1.09, 52, 7870],
                'N': [0.96, 14, 1250],
                'Si': [0.65, 28, 2330],
                'Mg': [0.58, 24, 1740],
                'S': [0.44, 32, 2070]}

# universe generation variables
planet_min_count = 15  # min number of generated bodies
planet_max_count = 15  # max number of bodies to generate
starting_point_max = .5 * AU  # max position in 3d space
big_bang_momentum = 0  # starting impulse for planets in kg*m/s

# universe simulation variables
timestep = 90  # measured in seconds
turns_to_flagcheck = 10  # steps to take before a planet checks whether intensive calculations are needed (unused)
proximity_limit = 1 * AU  # radius to check for proximity alert on object simulation
turn_limit = 2000  # how long to run the simulation
alert_freq = 50  # print status every N turns
use_quadrants = False  # use quadrants to optimize calculations. currently used for multiproccessing sims
resolve_collisions = True  # to make prototyping easier, ignore collisions
use_custom_seed = False  # look for user specified seed when initializing
quiet = True  # whether we should spam the console when plotting (will update later with levels)
generate_plots = False  # whether we should output plots
generate_json = True

# json file variables
dist_units = 15 * solar_radius  # can't have such huge numbers, so all values will be in solar radii

# log file initialization
# two files - one for generation, one for simulation
now = datetime.datetime.now()
log_path = 'logs'
runtime_dir = str(now.day) + str(now.month) + str(now.year) + str(now.hour)
work_dir = os.path.join(log_path, runtime_dir)
try:
    os.mkdir(work_dir)
except FileExistsError:
    pass
gen_file_name = 'generation.txt'
sim_file_name = 'simulation.txt'
image_name = 'output.png'
gen_path = os.path.join(work_dir, gen_file_name)
sim_path = os.path.join(work_dir, sim_file_name)
image_path = os.path.join(work_dir, image_name)
gen_log_file = open(gen_path, 'w', encoding="utf-8")
sim_log_file = open(sim_path, 'w', encoding="utf-8")

# seed generation
if use_custom_seed:
    seed = input('Enter seed: ')
    try:
        seed = int(seed)
    except ValueError:
        print("Seed must be an integer. Generate random seed?")
        cont = input('y/n: ')
        if cont == 'y':
            seed = random.randint(0, 999999)
        else:
            exit()

else:
    seed = random.randint(0, 999999)
random.seed(seed)
np.random.seed(seed)
gen_log_file.writelines('Seed for this execution: ' + str(seed) + '\n')
sim_log_file.writelines('Seed for this execution: ' + str(seed) + '\n')

# random name generation setup
name_generator = MarkovGenerator()
file_dir = 'name_data'
file_list = ['canada_place_names.csv', 'city_names.csv', 'hygdata_v3.csv']
for file in file_list:
    file_path = os.path.join(file_dir, file)
    name_generator.load_csv(file_path)

# parallel processing initialization
thread_count = 8  # how many threads to utilize (4 on laptop, 8 on pc)


class Universe:
    def __init__(self):
        self.name = name_generator.generate().title()
        self.total_contents = {}
        self.active_contents = {}
        self.removed_contents = {}
        self.collision_points = []
        self.number_of_bodies = random.randint(planet_min_count, planet_max_count)
        gen_log_file.writelines('Number of bodies to generate: ' + str(self.number_of_bodies) + '\n')
        self.stars = []
        self.planets = []
        self.blackholes = []
        if use_quadrants:
            self.quadrants = {(1, 1, 1): [],
                              (1, 1, 0): [],
                              (1, 0, 1): [],
                              (1, 0, 0): [],
                              (0, 1, 1): [],
                              (0, 1, 0): [],
                              (0, 0, 1): [],
                              (0, 0, 0): []}
        self.color_dict = {'star': 'magma',
                           'planet': 'winter',
                           'blackhole': 'binary'}
        self.up_time = 0
        self.vpy_spheres = []
        self.colliding_bodies = []
        for i in range(self.number_of_bodies):
            self.generate_planet()
        for body in self.active_contents:
            body.get_surface_temp(self)
            if body.is_blackhole:
                self.blackholes.append(body)
            elif body.can_fuse:
                self.stars.append(body)
            else:
                self.planets.append(body)
        gen_log_file.writelines('Generated %d planets, %d stars, and %d blackholes' % (
            len(self.planets), len(self.stars), len(self.blackholes)) + '\n')
        print('Generation complete - number of stars: %d of %d' % (len(self.stars), self.number_of_bodies))
        gen_log_file.close()
        if generate_plots:
            print('Initializing Plots')
            self.output = plt.figure()
            self.ax1 = self.output.add_subplot(2, 2, 1, projection="3d")
            self.ax2 = self.output.add_subplot(2, 2, 2)
            self.ax3 = self.output.add_subplot(2, 2, 3)
            self.ax4 = self.output.add_subplot(2, 2, 4)
            self.ax1.set_title('3D')
            self.ax2.set_title('Overhead')
            self.ax3.set_title('Front')
            self.ax4.set_title('Side')
            self.ax1.set_xlabel('X')
            self.ax1.set_ylabel('Y')
            self.ax2.set_xlabel('X')
            self.ax2.set_ylabel('Y')
            self.ax3.set_xlabel('X')
            self.ax3.set_ylabel('Z')
            self.ax4.set_xlabel('Y')
            self.ax4.set_ylabel('Z')
        if generate_json:
            print('Initializing JSON')
            json_path = 'xperience'
            json_name = 'lastgen.json'
            json_path = os.path.join(json_path, json_name)
            self.json_file = open(json_path, 'w')
            self.json_output = ['{"name": "%s", "turns": [' % self.name]

    def generate_planet(self):
        print('Generating Body')
        body_comp = self.generate_composition()
        new_body = CelestialBody(body_comp, self)
        self.active_contents[new_body] = []

    def dump_info(self):
        lines = ['{"name": "%s", "turns": [' % self.name]
        self.total_contents = self.active_contents.copy()
        self.total_contents.update(self.removed_contents)
        for turn in range(turn_limit):
            lines.append('{"planets": [')
            for body in self.total_contents:
                timestamp, position = self.total_contents[body][turn]
                line = '{"name": "%s", "radius": "%f", "position": ["%f", "%f", "%f"],' % (
                    body.name, body.radius / dist_units, position[0] / dist_units, position[1] / dist_units,
                    position[2] / dist_units)
                if timestamp == turn:
                    line += ' "exists": "true"},'
                else:
                    line += ' "exists": "false"},'
                lines.append(line)
            lines[-1] = lines[-1][:-1]  # remove trailing comma from list
            lines.append(']},')
        lines[-1] = lines[-1][:-1]
        lines.append(']}')
        self.json_file.writelines(lines)
        self.json_file.close()

    def plot_data(self, data='mass'):
        if data == 'mass':
            plot = [body.mass for body in self.active_contents]
            plot.sort()
            plt.hist(plot, 100)
            plt.xlabel('Mass (kg)')
            plt.show()

    def get_stars(self):
        return self.stars

    def get_planets(self):
        return self.planets

    def update_step(self):
        """Four phases to an update: First phase is promises. Pending property changes are calculated here. Second
        phase calculates all the changes for the planet for one timestep (updating pos, v, a). Third phase applies
        the changes. Fourth phase is a promise check. If any bodies send up a flag signalling more thorough
        calculations needed, they are added to the promise list for next turn. """
        sim_log_file.write('\nExecuting step %d: \n' % self.up_time)
        if len(self.colliding_bodies) > 0:
            sim_log_file.write('There are %d colliding systems this step\n' % len(self.colliding_bodies))
            map(self.resolve_collisions, self.colliding_bodies)
            self.colliding_bodies = []
        if generate_json:
            json_file = self.json_output
            json_file.append('{"planets": [')
            append = json_file.append
            for key in self.active_contents:
                self.calculate_body_update(key)
                self.apply_body_update(key)
                self.check_body_flags(key)
                position = key.position
                append('{"name": "%s", "radius": "%f", "position": ["%f", "%f", "%f"], "exists": "true"},' % (
                    key.name, key.radius / dist_units, position[0] / dist_units, position[1] / dist_units,
                    position[2] / dist_units))
            for key in self.removed_contents:
                position = key.position
                append('{"name": "%s", "radius": "%f", "position": ["%f", "%f", "%f"], "exists": "false"},' % (
                    key.name, key.radius / dist_units, position[0] / dist_units, position[1] / dist_units,
                    position[2] / dist_units))
            json_file[-1] = json_file[-1][:-1]
            append(']},')
            self.json_output = json_file
        else:
            for key in self.active_contents:
                self.calculate_body_update(key)
                self.apply_body_update(key)
                self.check_body_flags(key)
        self.up_time += 1
        sim_log_file.write('Finished step\n')

    def calculate_body_update(self, body):
        body.calculate_update(self)

    def check_body_flags(self, body):
        self.active_contents[body].append((self.up_time, body.position))
        if len(body.flags) > 0:
            sim_log_file.write(str(body) + 'has %d flags after calculations\n' % len(body.flags))
            body.promised = True
            if type(body.flags[0]) == list:
                for other in body.flags[0]:
                    sim_log_file.write(str(body) + 'is colliding with ' + str(other) + '\n')
                self.register_colliding_bodies(body.flags[0])
                del body.flags[0]

    @staticmethod
    def apply_body_update(body):
        body.apply_update()

    def register_colliding_bodies(self, system):
        system = sorted(system, key=lambda x: str(x))
        if system not in self.colliding_bodies:
            self.colliding_bodies.append(system)

    @staticmethod
    def check_quadrant_promise(quadrant):
        for body in quadrant:
            if body.promised:
                return True
        return False

    @staticmethod
    def generate_composition():
        unaccounted_matter = 100
        composition = [['H', 0], ['He', 0], ['O', 0], ['C', 0], ['Ne', 0], ['Fe', 0], ['N', 0], ['Si', 0], ['Mg', 0],
                       ['S', 0]]
        while unaccounted_matter > 0:
            for i in range(len(composition)):
                if unaccounted_matter > 0:
                    element = composition[i][0]
                    component = element_dict[element]
                    abundance = component[0] * random.randint(500, 1500) / 1000
                    abundance = min(unaccounted_matter, abundance)
                    abundance = max(abundance, 0)
                    composition[i][1] += abundance
                    unaccounted_matter -= abundance
        assert unaccounted_matter == 0
        return composition

    def resolve_collisions(self, workspace):
        sim_log_file.write('Resolving collisions for:\n')
        total_mass = 0
        total_p = 0
        total_l = 0
        total_composition = [[x, 0] for x in element_dict.keys()]
        total_composition = sorted(total_composition, key=lambda x: x[0])
        total_position = [0, 0, 0]
        for body in workspace:
            sim_log_file.write('\t%s\n' % body)
            total_mass += body.mass
            total_p += (body.mass * body.velocity)
            mom_iner = .4 * body.mass * body.radius ** 2
            total_l += (mom_iner * body.spin)
            for index, pair in enumerate(body.composition):
                total_composition[index][1] += pair[1]
            total_position += body.position
            self.removed_contents[body] = self.active_contents[body]
            del self.active_contents[body]
        new_mass = total_mass
        new_velocity = total_p / new_mass
        for element, abundance in total_composition:
            abundance /= len(workspace)
        new_composition = total_composition
        new_position = total_position / len(workspace)
        new_body = CelestialBody(new_composition, self, new_mass, new_position, new_velocity, total_l,
                                 time=self.up_time)
        self.active_contents[new_body] = []
        self.active_contents[new_body].append((self.up_time, new_body.position))
        self.collision_points.append(new_position)

    def reesolve_collisions(self, workspace):
        body1 = workspace.pop(0)
        body2 = workspace.pop(1)
        if body1.mass < body2.mass:
            projectile = body1
            target = body2
        else:
            target = body1
            projectile = body2
        cstar = 5  # material constant for how energy is dissipated, higher means sharper? 5 for solids, 1.9 for fluids
        total_mass = body1.mass + body2.mass
        average_density = (body1.density + body2.density) / 2
        mu = random.randrange(333, 666) / 1000  # material constant for some physics shit, falls between .33 and .66
        gamma = projectile.mass / total_mass  # ratio of projectile mass to larger mass (smaller body is projectile)
        target_radius = target.radius * 100  # radius of larger body in cm
        system_velocity = (target.velocity - projectile.velocity)
        impact_velocity = np.sqrt(system_velocity.dot(system_velocity)) * 100  # velocity at impact in cm/s
        impact_angle = self.get_vector_angles(body1.velocity, body2.velocity)  # 0 rad = head on, pi/2 = perpendicular
        bb = np.sin(impact_angle)  # not sure why, but physics wants impact angle in sin(x) radians
        ## HERE BE DRAGONS, I DONT UNDERSTAND THEIR SHITTY VAR NAMES ##
        rho1 = 1  # density of one body? 1 g/cm3. not really sure
        barr = [
        0.0000000,
        0.045693737,
        0.066364088,
        0.082256556,
        0.095746684,
        0.10747497,
        0.11814432,
        0.12797298,
        0.13711830,
        0.14570045,
        0.15381270,
        0.16152855,
        0.16890663,
        0.17599424,
        0.18282998,
        0.18938688,
        0.19572389,
        0.20189088,
        0.20790679,
        0.21368939,
        0.21935568,
        0.22491598,
        0.23027924,
        0.23556216,
        0.24073394,
        0.24577314,
        0.25075642,
        0.25559341,
        0.26037714,
        0.26506051,
        0.26966749,
        0.27420728,
        0.27865712,
        0.28306245,
        0.28737215,
        0.29165130,
        0.29583558,
        0.29999617,
        0.30406782,
        0.30811682,
        0.31208701,
        0.31603082,
        0.31990928,
        0.32375382,
        0.32754908,
        0.33129978,
        0.33501932,
        0.33868123,
        0.34233159,
        0.34590940,
        0.34948218,
        0.35299439,
        0.35648750,
        0.35994531,
        0.36336282,
        0.36677042,
        0.37011604,
        0.37345687,
        0.37675438,
        0.38002695,
        0.38328439,
        0.38649197,
        0.38969489,
        0.39285761,
        0.39599871,
        0.39912906,
        0.40221124,
        0.40528885,
        0.40833698,
        0.41135844,
        0.41437538,
        0.41734794,
        0.42031136,
        0.42326098,
        0.42617324,
        0.42908110,
        0.43196420,
        0.43482324,
        0.43767790,
        0.44049949,
        0.44330752,
        0.44611121,
        0.44887575,
        0.45163489,
        0.45438477,
        0.45710128,
        0.45981351,
        0.46251225,
        0.46518369,
        0.46785091,
        0.47050190,
        0.47313010,
        0.47575408,
        0.47836041,
        0.48094704,
        0.48352948,
        0.48609393,
        0.48864063,
        0.49118315,
        0.49370823,
        0.49621650,
        0.49872064,
        0.50120863,
        0.50367995,
        0.50614716,
        0.50860012,
        0.51103588,
        0.51346757,
        0.51588737,
        0.51828889,
        0.52068634,
        0.52307465,
        0.52544320,
        0.52780768,
        0.53016602,
        0.53250281,
        0.53483555,
        0.53716421,
        0.53947146,
        0.54177362,
        0.54407170,
        0.54635266,
        0.54862530,
        0.55089390,
        0.55314963,
        0.55539383,
        0.55763399,
        0.55986545,
        0.56208222,
        0.56429495,
        0.56650301,
        0.56869331,
        0.57087961,
        0.57306184,
        0.57522978,
        0.57739052,
        0.57954723,
        0.58169408,
        0.58383018,
        0.58596224,
        0.58808859,
        0.59020090,
        0.59230918,
        0.59441341,
        0.59650489,
        0.59859020,
        0.60067147,
        0.60274416,
        0.60480729,
        0.60686637,
        0.60892066,
        0.61096237,
        0.61300003,
        0.61503361,
        0.61705726,
        0.61907422,
        0.62108710,
        0.62309367,
        0.62509063,
        0.62708351,
        0.62907230,
        0.63105082,
        0.63302437,
        0.63499382,
        0.63695630,
        0.63891117,
        0.64086193,
        0.64280850,
        0.64474533,
        0.64667804,
        0.64860660,
        0.65052818,
        0.65244342,
        0.65435450,
        0.65626094,
        0.65815931,
        0.66005350,
        0.66194351,
        0.66382691,
        0.66570478,
        0.66757844,
        0.66944735,
        0.67130945,
        0.67316731,
        0.67502093,
        0.67686854,
        0.67871111,
        0.68054943,
        0.68238306,
        0.68421087,
        0.68603438,
        0.68785360,
        0.68966751,
        0.69147672,
        0.69328163,
        0.69508198,
        0.69687736,
        0.69866838,
        0.70045504,
        0.70223711,
        0.70401472,
        0.70578794,
        0.70755681,
        0.70932146,
        0.71108168,
        0.71283745,
        0.71458937,
        0.71633702,
        0.71808018,
        0.71981921,
        0.72155469,
        0.72328565,
        0.72501208,
        0.72673546,
        0.72845461,
        0.73016920,
        0.73187992,
        0.73358766,
        0.73529078,
        0.73698927,
        0.73868548,
        0.74037751,
        0.74206485,
        0.74374871,
        0.74543001,
        0.74710657,
        0.74877840,
        0.75044889,
        0.75211502,
        0.75377634,
        0.75543476,
        0.75709077,
        0.75874193,
        0.76038824,
        0.76203441,
        0.76367572,
        0.76531212,
        0.76694649,
        0.76857830,
        0.77020509,
        0.77182760,
        0.77345013,
        0.77506760,
        0.77668001,
        0.77829177,
        0.77990021,
        0.78150352,
        0.78310375,
        0.78470344,
        0.78629791,
        0.78788716,
        0.78947778,
        0.79106367,
        0.79264426,
        0.79422374,
        0.79580129,
        0.79737346,
        0.79894181,
        0.80051126,
        0.80207524,
        0.80363373,
        0.80519406,
        0.80675006,
        0.80830052,
        0.80985020,
        0.81139843,
        0.81294098,
        0.81448010,
        0.81602074,
        0.81755561,
        0.81908469,
        0.82061748,
        0.82214484,
        0.82366630,
        0.82518913,
        0.82670913,
        0.82822312,
        0.82973612,
        0.83124891,
        0.83275558,
        0.83425892,
        0.83576463,
        0.83726409,
        0.83875797,
        0.84025672,
        0.84174908,
        0.84323504,
        0.84472562,
        0.84621100,
        0.84768981,
        0.84917178,
        0.85065019,
        0.85212191,
        0.85359555,
        0.85506708,
        0.85653175,
        0.85799738,
        0.85946206,
        0.86091971,
        0.86237768,
        0.86383551,
        0.86528614,
        0.86673682,
        0.86818779,
        0.86963137,
        0.87107520,
        0.87251926,
        0.87395573,
        0.87539315,
        0.87683023,
        0.87825952,
        0.87969102,
        0.88112101,
        0.88254302,
        0.88396913,
        0.88539189,
        0.88680641,
        0.88822768,
        0.88964304,
        0.89105046,
        0.89246693,
        0.89387465,
        0.89527819,
        0.89668704,
        0.89808686,
        0.89948719,
        0.90088811,
        0.90227971,
        0.90367753,
        0.90507016,
        0.90645534,
        0.90784921,
        0.90923313,
        0.91061710,
        0.91200212,
        0.91337681,
        0.91476039,
        0.91613600,
        0.91750828,
        0.91888493,
        0.92025048,
        0.92162388,
        0.92299017,
        0.92435325,
        0.92572034,
        0.92707543,
        0.92844135,
        0.92979680,
        0.93115362,
        0.93250939,
        0.93385757,
        0.93521360,
        0.93655606,
        0.93790983,
        0.93925182,
        0.94059851,
        0.94193981,
        0.94328007,
        0.94462039,
        0.94595493,
        0.94729390,
        0.94862350,
        0.94996064,
        0.95128617,
        0.95262090,
        0.95394320,
        0.95527478,
        0.95659482,
        0.95792238,
        0.95924112,
        0.96056358,
        0.96188201,
        0.96319804,
        0.96451711,
        0.96582512,
        0.96714572,
        0.96845341,
        0.96976664,
        0.97107799,
        0.97237832,
        0.97369359,
        0.97499968,
        0.97629708,
        0.97760891,
        0.97891370,
        0.98021245,
        0.98151163,
        0.98281788,
        0.98412057,
        0.98542089,
        0.98672016,
        0.98801963,
        0.98932040,
        0.99062314,
        0.99192764,
        0.99323206,
        0.99453230,
        0.99585543,
        0.99717836,
        0.99851871]  # some weird-ass probability list
        rhon = average_density / 1000  # density of one body? site says both bodies assumed same rho, so who knows
        mt = (4 * math.pi * rhon * target_radius ** 3) / 3
        mp = gamma * mt
        mtot = (1 + gamma) * mt
        mred = mp * mt / mtot
        rpn = (mp / (4 * math.pi * rhon / 3)) ** (1/3)
        rc1 = (mt * (1 + gamma) / (4 * math.pi * rhon / 3)) ** (1/3)
        vescrtn = (2 * G * mtot / (target_radius + rpn)) ** (1/2)
        vescrtn = (2 * G * mtot / target_radius) ** (1/2)
        vesc = vescrtn  # why? duplicate values for planets? maybe these should be different for bodies?
        qg = (1/8) * ((32 * math.pi * cstar / 5) ** (3/2 * mu))
        bcrit = target_radius / (target_radius + rpn)
        qangle = self.quangle  # this is some array len 400 fuck this shit
        qangle1 = self.quangle1  # because i thought their names couldnt get worse
        # determining outcome
        barrind = 0  # index of the impact parameter? not really sure
        for index in range(len(barr)):
            if bb > barr[index]:
                barrind = index
        if bb < bcrit:  # non-grazing regime
            gsim = (0.5 * (mp * mt / (mp + mt)) * impact_velocity * impact_velocity) / (mp + mt)
            mlrmtot = -0.5 * (qsim / qangle[])



    def __str__(self):
        return 'number of stars: %d of %d' % (len(self.stars), self.number_of_bodies)

    @staticmethod
    def get_vector_angles(vec1, vec2):
        uvec1 = vec1 / np.linalg.norm(vec1)
        uvec2 = vec2 / np.linalg.norm(vec2)
        angle = np.arccos(np.clip(np.dot(uvec1, uvec2), -1.0, 1.0))
        if angle > math.pi / 2:
            angle = math.pi - angle
        return angle

    def run_sim(self):
        print('Begin simulation')
        while self.up_time < turn_limit:
            if self.up_time % alert_freq == 0:
                print('Running step %d' % self.up_time)
            self.update_step()
        sim_log_file.close()
        print('Simulation Complete')
        if generate_json:
            body_count = len(self.active_contents) + len(self.removed_contents)
            self.json_output[-1] = self.json_output[-1][:-1]
            self.json_output.append('], "body_count": "%d"}' % body_count)
            self.json_file.writelines(self.json_output)
            self.json_file.close()
            print('JSON File finished - %d total bodies simulated' % body_count)
        if generate_plots:
            print('Making Plots')
            self.generate_sim_plot()

    def generate_sim_plot(self):
        mpl.rcParams['legend.fontsize'] = 6
        print('Getting total contents of universe')
        self.total_contents = self.active_contents.copy()
        self.total_contents.update(self.removed_contents)
        list(map(self.plot_body, self.total_contents))
        print('Plotting collisions')
        for point in self.collision_points:
            self.ax1.scatter(int(point[0]), int(point[1]), int(point[2]), c='r')
            self.ax2.scatter(int(point[0]), int(point[1]), c='r')
            self.ax3.scatter(int(point[0]), int(point[2]), c='r')
            self.ax4.scatter(int(point[1]), int(point[2]), c='r')
        print('Finishing up')
        plt.tight_layout()
        plt.savefig(image_path, dpi=3000)
        print('Plot generation complete')

    def plot_body(self, body):
        print('Working on %s' % body)
        cmap = self.color_dict[body.type]
        cm = plt.get_cmap(cmap)
        xs = []
        ys = []
        zs = []
        print('Getting history')
        for step, point in self.total_contents[body]:
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        self.ax1.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        self.ax2.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        self.ax3.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        self.ax4.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        print('Plotting history')
        for i in range(turn_limit - 1):
            if not quiet:
                print(i)
            self.ax1.plot(xs[i:i + 2], ys[i:i + 2], zs[i:i + 2], solid_capstyle='projecting')
            self.ax2.plot(xs[i:i + 2], ys[i:i + 2], solid_capstyle='projecting')
            self.ax3.plot(xs[i:i + 2], zs[i:i + 2], solid_capstyle='projecting')
            self.ax4.plot(ys[i:i + 2], zs[i:i + 2], solid_capstyle='projecting')
        print('Labelling')
        self.ax1.text(xs[-1], ys[-1], zs[-1], '%s' % body.name, size=5, color='b')

    @staticmethod
    def build_and_run():
        universe = Universe()
        universe.run_sim()


class BodyUpdate:
    def __init__(self, body):
        self.luminosity = body.luminosity
        self.surface_temperature = body.surface_temperature
        self.can_fuse = body.can_fuse
        self.position = body.position
        self.velocity = body.velocity
        self.acceleration = body.acceleration
        self.rotation = body.rotation


class CelestialBody:
    def __init__(self, composition, parent, mass=None, position=None, velocity=None, ang_mom=None, time=0):
        self.name = name_generator.generate().title()
        if mass is None:
            self.mass = np.random.binomial(1000, star_freq) / 1000 * mpz(9 * 10 ** 29)
        else:
            self.mass = mass
        self.can_fuse = False
        self.albedo = random.random()
        self.composition = composition
        self.density = self.get_avg_density()
        self.volume = self.mass / self.density
        self.radius = (3 * self.volume / (4 * math.pi)) ** (1 / 3)
        self.gravity = (G * self.mass) / (self.radius ** 2)
        self.core_pressure = 3 / (4 * math.pi * G) * self.gravity ** 2
        self.luminosity = self.get_luminosity()
        self.init_temp = self.get_surface_temp(parent, True)
        self.surface_temperature = None
        self.can_fuse = self.has_fusion(True)
        if position is None:
            self.position = np.random.uniform(-1, 1, 3) * starting_point_max
        else:
            self.position = position
        if velocity is None:
            self.velocity = self.big_bang()
        else:
            self.velocity = velocity
        if ang_mom is None:
            self.spin = np.random.randint(-starting_spin, starting_spin, 3)
        else:
            self.spin = ang_mom / (.4 * self.mass * self.radius ** 2)
        self.acceleration = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.escape_vel = (2 * self.gravity * self.radius) ** .5
        self.schwarzchild_radius = 2 * self.mass * G / (c ** 2)
        self.promised = True
        self.flags = []  # possible flags: blackhole formation, collision, proximity alert, nearing boundary
        self.turns_since_check = 0
        self.is_proximal = False
        self.is_blackhole = self.check_schwarz_radius()
        if self.is_blackhole:
            self.update_blackhole()
        self.type = self.get_type()
        self.previous_state = {'star': self.can_fuse,
                               'blackhole': self.is_blackhole,
                               'proximity_alert': self.is_proximal
                               }
        self.current_state = {'star': self.can_fuse,
                              'blackhole': self.is_blackhole,
                              'proximity_alert': self.is_proximal
                              }
        self.log_gen_info(time)
        self.update = None

    def check_schwarz_radius(self):
        if self.radius <= self.schwarzchild_radius:
            return True
        return False

    def __str__(self):

        string = self.type + ' ' + self.name + ' '
        return string

    def update_blackhole(self):
        self.albedo = 1
        self.luminosity = 0
        self.surface_temperature = 0
        self.luminosity = 0
        self.can_fuse = False

    def big_bang(self):
        unit_vector = self.get_distance()[0]
        velocity_mag = big_bang_momentum / self.mass
        velocity = velocity_mag * unit_vector
        return velocity

    def log_gen_info(self, time):
        info = []
        str1 = 'Generated a ' + self.type + ' at position: ' + str(self.position) + '\n'
        info.append(str1)
        attributes = vars(self)
        for attr in attributes:
            if attr == 'composition':
                for element, abundance in attributes[attr]:
                    string = '\t%s: %f\n' % (element, abundance)
                    info.append(string)
            else:
                string = '\t%s: ' % attr + str(attributes[attr]) + '\n'
                info.append(string)
        if time == 0:
            gen_log_file.writelines(info)
        else:
            sim_log_file.writelines(info)

    def has_fusion(self, init=False):
        if self.radius > 0.0435 * solar_radius:
            if self.luminosity > 1.25e-3 * solar_luminosity:
                if init:
                    if self.init_temp > 2000:
                        return True
                else:
                    if self.surface_temperature > 2000:
                        return True
        return False

    def get_surface_temp(self, parent, init=False):
        if init:
            kt = G * self.mass * 1 / (3 * self.radius)
            surf_temp = kt / (1000 * k)
            return surf_temp
        else:
            if self.can_fuse:
                kt = G * self.mass * 1 / (3 * self.radius)
                surf_temp = kt / (1000 * k)
            else:
                star_list = parent.get_stars()
                temp = 0
                for star in star_list:
                    dist = [(a - b) ** 2 for a, b in zip(self.position, star.position)]
                    dist = math.sqrt(sum(dist))
                    numerator = star.luminosity * (1 - self.albedo)
                    denominator = (16 * math.pi * s * dist ** 2)
                    if denominator > 0:
                        work_var = numerator / denominator
                    else:
                        work_var = 0
                    star_temp = work_var ** .25
                    temp += star_temp
                surf_temp = temp
            self.surface_temperature = surf_temp

    def get_luminosity(self):
        mass_ratio = self.mass / solar_mass
        if self.mass < 0.43 * solar_mass:
            luminosity = 0.23 * mass_ratio ** 2.3
        elif self.mass < 2 * solar_mass:
            luminosity = mass_ratio ** 4
        elif self.mass < 20 * solar_mass:
            luminosity = 1.5 * mass_ratio ** 3.5
        else:
            luminosity = 3200 * mass_ratio
        luminosity *= solar_luminosity
        return luminosity

    def check_quadrant_boundary(self):
        boundary_proximity_check = 1.25 * self.radius
        for dimension in self.position:
            if abs(dimension) < boundary_proximity_check:
                pass

    def quick_update(self, check_needed):
        self.position = np.add(self.position, np.multiply(self.velocity, timestep), out=self.position, casting="unsafe")
        self.rotation += self.spin * timestep
        if check_needed:
            self.run_flag_check()
            self.turns_since_check = 0
        else:
            self.turns_since_check += 1
        for key in self.previous_state:
            self.previous_state[key] = bool(self.current_state[key])

    def run_flag_check(self):
        for key in self.previous_state:
            if self.previous_state[key] != self.current_state[key]:
                self.flags.append(key)

    def calculate_update(self, parent):
        sim_log_file.writelines('%s\n' % self)
        current_quadrant = list(parent.active_contents.keys())
        nearby_objects = self.proximity_check(current_quadrant)
        self.update = BodyUpdate(self)
        if len(nearby_objects) > 0:
            sim_log_file.writelines('\tis now proximal\n')
            colliding_with = self.collision_check(nearby_objects)
            if colliding_with is not None:
                sim_log_file.writelines('\tis now colliding with: ')
                for body in colliding_with:
                    sim_log_file.writelines(str(body) + ' ')
                sim_log_file.writelines('\n')
                self.flags.insert(0, colliding_with)
        if not self.is_blackhole:
            self.update.luminosity = self.get_luminosity()
            self.update.surface_temperature = self.get_surface_temp(parent)
            self.update.can_fuse = self.has_fusion(True)
        self.update.position = np.add(self.position, np.multiply(self.velocity, timestep), casting="unsafe")
        self.update.velocity = np.add(self.velocity, np.multiply(self.acceleration, timestep), casting="unsafe")
        sim_log_file.write('\tis now located at: ' + str(self.position) + '\n')
        sim_log_file.write('\tis now moving at ' + str(self.velocity) + ' m/dt\n')
        self.update.acceleration = self.update_acceleration(nearby_objects)
        sim_log_file.write('\tis now accelerating at ' + str(self.velocity) + ' m/dt^2\n')
        self.update.rotation += self.spin * timestep

    def apply_update(self):
        assert self.update is not None
        self.luminosity = self.update.luminosity
        self.surface_temperature = self.update.surface_temperature
        self.can_fuse = self.update.can_fuse
        self.position = self.update.position
        self.velocity = self.update.velocity
        self.acceleration = self.update.acceleration
        self.rotation = self.update.rotation
        self.run_flag_check()
        for key in self.previous_state:
            self.previous_state[key] = bool(self.current_state[key])
        self.update = None
        return

    def massless_update(self, parent, current_quadrant=None):
        sim_log_file.writelines('%s\n' % self)
        if current_quadrant is None:
            current_quadrant = list(parent.active_contents.keys())
        nearby_objects = self.proximity_check(current_quadrant)
        if len(nearby_objects) > 0:
            sim_log_file.writelines('\tis now proximal\n')
            colliding_with = self.collision_check(nearby_objects)
            self.is_proximal = True
            if colliding_with is not None:
                sim_log_file.writelines('\tis now colliding with: ')
                for body in colliding_with:
                    sim_log_file.writelines(str(body) + ' ')
                sim_log_file.writelines('\n')
                self.flags.insert(0, colliding_with)
        if not self.is_blackhole:
            self.luminosity = self.get_luminosity()
            self.surface_temperature = self.get_surface_temp(parent)
            self.can_fuse = self.has_fusion(True)
        self.position = np.add(self.position, np.multiply(self.velocity, timestep), casting="unsafe")
        self.velocity = np.add(self.velocity, np.multiply(self.acceleration, timestep), casting="unsafe")
        sim_log_file.write('\tis now located at: ' + str(self.position) + '\n')
        sim_log_file.write('\tis now moving at ' + str(self.velocity) + ' m/dt\n')
        self.acceleration = self.update_acceleration(nearby_objects)
        sim_log_file.write('\tis now accelerating at ' + str(self.velocity) + ' m/dt^2\n')
        self.rotation += self.spin * timestep
        self.run_flag_check()
        self.turns_since_check = 0
        for key in self.previous_state:
            self.previous_state[key] = bool(self.current_state[key])
        return

    def collision_check(self, bodies):
        colliding_with = []
        for body, distance in bodies:
            if body is not self:
                other_radius = body.radius
                if distance <= self.radius + other_radius:
                    colliding_with.append(body)
        if len(colliding_with) > 0:
            colliding_with.append(self)
            return colliding_with
        return None

    def proximity_check(self, quadrant):
        proximal_bodies = []
        for body in quadrant:
            if body is not self:
                distance = self.get_distance(body)[1]
                if distance < self.radius + body.radius:
                    pass
                if distance < proximity_limit:
                    proximal_bodies.append((body, distance))
        return proximal_bodies

    def get_avg_density(self):
        density_sum = 0
        density_components = len(self.composition)

        abundance_sum = 0
        for element, abundance in self.composition:
            abundance_sum += abundance
            unit_density = element_dict[element][2]
            weighted_density = unit_density * abundance / 6.79447
            density_sum += weighted_density
        avg_density = density_sum / density_components
        if avg_density <= 0:
            print('avg', avg_density)
            print('sum', density_sum)
        assert avg_density > 0
        return avg_density

    def get_distance(self, point=None):
        if point is None:
            point = [0, 0, 0]
        else:
            point = point.position
        direction_vector = point - self.position
        magnitude = (direction_vector.dot(direction_vector)) ** .5
        unit_vector = direction_vector / magnitude
        return unit_vector, magnitude

    def update_acceleration(self, nearby_objects):
        acceleration = self.acceleration
        for body, distance in nearby_objects:
            direction, distance = self.get_distance(body)
            grav_acc = G * body.mass / (distance ** 2)
            da = grav_acc * direction
            acceleration += da
        return acceleration

    def get_type(self):
        if self.is_blackhole:
            return 'blackhole'
        elif self.can_fuse:
            return 'star'
        return 'planet'

    def __repr__(self):
        return str(self)


def stress_test():
    global turn_limit, seed, timestep, alert_freq, quiet, thread_count, planet_min_count, planet_max_count, use_quadrants, generate_json, generate_plots, test_map
    turn_limit = 2000
    seed = 1000
    timestep = 30
    alert_freq = 100
    quiet = False
    use_quadrants = True
    thread_count = 4
    planet_count = 8
    planet_max_count = planet_count
    planet_min_count = planet_count
    generate_json = True
    generate_plots = True
    random.seed(seed)
    np.random.seed(seed)
    testverse = Universe()
    testverse.run_sim()


Universe.build_and_run()
