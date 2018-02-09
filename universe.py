import os.path
import math
import random
import datetime
import timeit

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from gmpy2 import mpz
from mpl_toolkits.mplot3d import Axes3D
from markovnamegen import MarkovGenerator
from multiprocessing.dummy import Pool as ThreadPool

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
timestep = 180  # measured in seconds
turns_to_flagcheck = 10  # steps to take before a planet checks whether intensive calculations are needed (unused)
proximity_limit = 2 * AU  # radius to check for proximity alert on object simulation
turn_limit = 2000  # how long to run the simulation
alert_freq = 5  # print status every N turns
use_quadrants = False  # use quadrants to optimize calculations. currently broken
resolve_collisions = False  # to make prototyping easier, ignore collisions
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
thread_count = 2  # how many threads to utilize (4 on laptop, 8 on pc)


# results = pool.map(urllib2.urlopen, urls)
# close the pool and wait for the work to finish
# pool.close()
# pool.join()


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
        initpool = ThreadPool(thread_count)
        for i in range(self.number_of_bodies):
            initpool.apply_async(self.generate_planet(), ())
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
            plot = []
            for body in self.active_contents:
                plot.append(body.mass)
            plot.sort()
            plt.hist(plot, 100)
            plt.xlabel('Mass (kg)')
            plt.show()

    def get_stars(self):
        return self.stars

    def get_planets(self):
        return self.planets

    def update_step(self):
        """Three phases to an update: First phase is promises. Pending property changes are calculated here.
        Second phase updates all bodies. If the area had no promises, this is just updating position and rotation.
        Third phase is a promise check. If any bodies send up a flag signalling more thorough calculations needed, they
        are added to the promise list for next turn."""
        sim_log_file.write('\nExecuting step %d: \n' % self.up_time)
        promised_quadrants = []
        quick_quadrants = []
        planet_calculations = 0
        if len(self.colliding_bodies) > 0:
            sim_log_file.write('There are %d colliding systems this step\n' % len(self.colliding_bodies))
            self.resolve_collisions()
        if use_quadrants:
            for key in self.quadrants:
                sim_log_file.write('Checking quadrant ' + str(key) + '\n')
                quadrant = self.quadrants[key]
                if self.check_quadrant_promise(quadrant):
                    sim_log_file.write('\tThis quadrant was promised\n')
                    promised_quadrants.append(quadrant)
                else:
                    sim_log_file.write("\tThis quadrant wasn't promised\n")
                    quick_quadrants.append(quadrant)
            for index in range(len(promised_quadrants)):
                sim_log_file.write('Working on promises for quadrant %d of %d\n' % (index, len(promised_quadrants)))
                work_quadrant = promised_quadrants[index]
                for body in work_quadrant:
                    body.massless_update(self, work_quadrant)
                    self.active_contents[body].append((self.up_time, body.position))
                    if len(body.flags) > 0:
                        sim_log_file.write(str(body) + 'has %d flags after calculations\n' % len(body.flags))
                        body.promised = True
                        if type(body.flags[0]) == list:
                            for other in body.flags[0]:
                                sim_log_file.write(str(body) + 'is colliding with ' + str(other) + '\n')
                            self.colliding_bodies.append(body.flags[0])
                            del body.flags[0]
                    planet_calculations += 1
            for quadrant in quick_quadrants:
                sim_log_file.write('Running quick calculations on quadrant ' + str(quadrant) + '\n')
                for body in quadrant:
                    turns_to_check = turns_to_flagcheck - body.turns_since_check
                    if turns_to_check == 0:
                        check = True
                    else:
                        check = False
                    body.quick_update(check)
                    self.active_contents[body].append((self.up_time, body.position))
                    if check:
                        if len(body.flags) > 0:
                            sim_log_file.write(str(body) + 'has %d flags after calculations\n' % len(body.flags))
                            body.promised = True
                    planet_calculations += 1
        else:
            for key in self.active_contents:
                key.massless_update(self)
                self.active_contents[key].append((self.up_time, key.position))
                if len(key.flags) > 0:
                    sim_log_file.write(str(key) + 'has %d flags after calculations\n' % len(key.flags))
                    key.promised = True
                    if type(key.flags[0]) == list:
                        for other in key.flags[0]:
                            sim_log_file.write(str(key) + 'is colliding with ' + str(other) + '\n')
                        self.register_colliding_bodies(key.flags[0])
                        del key.flags[0]
                planet_calculations += 1
        self.up_time += 1
        assert planet_calculations == len(self.active_contents)  # make sure that all bodies have been updated
        sim_log_file.write('Finished step\n')

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

    def resolve_collisions(self):
        if resolve_collisions:
            while len(self.colliding_bodies) > 0:
                sim_log_file.write('Resolving collisions for:\n')
                workspace = self.colliding_bodies[0]
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
                self.active_contents[new_body].append(new_body.position)
                self.collision_points.append(new_position)
                del self.colliding_bodies[0]
        else:
            self.colliding_bodies = []

    def __str__(self):
        return 'number of stars: %d of %d' % (len(self.stars), self.number_of_bodies)

    def run_sim(self):
        print('Begin simulation')
        while self.up_time < turn_limit:
            if self.up_time % alert_freq == 0:
                print('Running step %d' % self.up_time)
            self.update_step()
        sim_log_file.close()
        print('Simulation Complete')
        if generate_plots:
            print('Making Plots')
            self.generate_sim_plot()
        if generate_json:
            print('Writing JSON')
            self.dump_info()

    def generate_sim_plot(self):
        pool = ThreadPool(thread_count)
        mpl.rcParams['legend.fontsize'] = 6
        print('Getting total contents of universe')
        self.total_contents = self.active_contents.copy()
        self.total_contents.update(self.removed_contents)
        pool.map(self.plot_body, self.total_contents)
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()
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
        for point in self.total_contents[body]:
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        self.ax1.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        self.ax2.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        self.ax3.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        self.ax4.set_prop_cycle('color', [cm(1. * i / (turn_limit - 1)) for i in range(turn_limit - 1)])
        print('Plotting history')
        for i in range(turn_limit - 1):
            self.ax1.plot(xs[i:i + 2], ys[i:i + 2], zs[i:i + 2], solid_capstyle='projecting')
            if not quiet:
                print('.')
            self.ax2.plot(xs[i:i + 2], ys[i:i + 2], solid_capstyle='projecting')
            if not quiet:
                print('..')
            self.ax3.plot(xs[i:i + 2], zs[i:i + 2], solid_capstyle='projecting')
            if not quiet:
                print('...')
            self.ax4.plot(ys[i:i + 2], zs[i:i + 2], solid_capstyle='projecting')
            if not quiet:
                print('..')
        print('Labelling')
        self.ax1.text(xs[-1], ys[-1], zs[-1], '%s' % body.name, size=5, color='b')
        #  self.ax1.quiver(xs[-1], ys[-1], zs[-1], body.acceleration[0], body.acceleration[1], body.acceleration[2],
        #             length=2e4)


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

    def get_distance(self, point=None, direction="away"):
        if point is None:
            point = [0, 0, 0]
        else:
            point = point.position
        if direction == 'towards':
            direction_vector = self.position - point
        else:
            direction_vector = point - self.position
        magnitude = (direction_vector.dot(direction_vector)) ** .5
        unit_vector = direction_vector / magnitude
        return unit_vector, magnitude

    def update_acceleration(self, nearby_objects):
        acceleration = self.acceleration
        for body, distance in nearby_objects:
            direction, distance = self.get_distance(body,
                                                    'tords')  # NOTE: Unit vector is maybe provided away from position
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
    global turn_limit, seed, timestep, alert_freq, quiet, thread_count, planet_min_count, planet_max_count
    turn_limit = 1000
    seed = 100
    timestep = 1
    alert_freq = 1000
    quiet = True
    thread_count = 2
    planet_count = 4
    planet_max_count = planet_count
    planet_min_count = planet_count
    random.seed(seed)
    np.random.seed(seed)
    testverse = Universe()
    testverse.run_sim()


def exec_stress_test():
    trial_count = 10
    exec_time = timeit.timeit(stmt=stress_test, number=trial_count)
    print(exec_time / trial_count)
    # laptop (1) : 279.58
    # laptop (2) :94.69
    # laptop (4) : 122.85
    # pc (1) :
    # pc (2) :
    # pc (4) :
    # pc (8) :


universe = Universe()
universe.run_sim()
