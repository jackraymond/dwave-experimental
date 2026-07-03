# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import unittest
import unittest.mock
from importlib.resources import files

import numpy

from dwave.cloud import Client, Solver
from dwave.cloud.exceptions import SolverNotFoundError
from dwave.cloud.testing.mocks import structured_solver_data
from dwave.system import DWaveSampler
from dwave.experimental.fast_reverse_anneal import (
    get_parameters, get_solver_name, SOLVER_FILTER,
    load_schedules, c_vs_t, plot_schedule,
)


class FRA(unittest.TestCase):

    def tearDown(self):
        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def test_sampler_params(self):
        x_target_c_range = [0, 1]
        x_nominal_pause_time_values = [0, 1, 2]
        info = ['fastReverseAnnealTargetCRange', x_target_c_range,
                'fastReverseAnnealNominalPauseTimeValues', x_nominal_pause_time_values]

        with unittest.mock.MagicMock() as sampler:
            sampler.solver.edges = [(0,1)]
            sampler.solver.sample_qubo.return_value.result.return_value = \
                dict(x_get_fast_reverse_anneal_exp_feature_info=info)

            p = get_parameters(sampler)

            self.assertIn('x_target_c', p)
            self.assertEqual(p['x_target_c']['limits']['range'], x_target_c_range)

            self.assertIn('x_nominal_pause_time', p)
            self.assertEqual(p['x_nominal_pause_time']['limits']['set'], x_nominal_pause_time_values)

    @unittest.mock.patch('dwave.experimental.fast_reverse_anneal.api.Client')
    def test_default_solver_name(self, client):
        class Solver:
            name = "mock-solver"

        client.from_config.return_value.__enter__.return_value.get_solver.return_value = Solver()

        solver_name = get_solver_name()

        self.assertEqual(solver_name, Solver.name)

    def test_schedules_smoke(self):
        fra = files('dwave.experimental.fast_reverse_anneal')
        schedules = json.loads(fra.joinpath('data/schedules.json').read_bytes())
        self.assertGreater(len(schedules.keys()), 0)

        with self.subTest('load_schedules'):
            solver_name = schedules.popitem()[0]
            schedules = load_schedules(solver_name=solver_name)
            self.assertIsInstance(schedules, dict)

        with self.subTest('c_vs_t'):
            t = numpy.array([0, 0.1, 0.2])
            c = c_vs_t(t, target_c=0, schedules=schedules)
            self.assertEqual(len(c), len(t))

        with self.subTest('plot_schedule'):
            plot_schedule(t, target_c=0, schedules=schedules)

    def test_schedule_regex_search(self):
        mock_schedules = {
            "Solver1(\\.\\d+)?": {
                "date": "...",
                "params": [{"nominal_pause_time": 0.0}]
            }
        }

        with unittest.mock.patch('dwave.experimental.fast_reverse_anneal.schedule._get_schedules_data',
                                 return_value=mock_schedules):
            self.assertIsInstance(load_schedules("Solver1"), dict)
            self.assertIsInstance(load_schedules("Solver1.1"), dict)
            self.assertIsInstance(load_schedules("Solver1.23"), dict)

            with self.assertRaises(ValueError):
                self.assertIsInstance(load_schedules("Solver1.2.3"), dict)
            with self.assertRaises(ValueError):
                load_schedules("Solver2")
            with self.assertRaises(ValueError):
                load_schedules("Solver10")


class SolverSelection(unittest.TestCase):

    def setUp(self):
        self.client = Client(endpoint='endpoint', token='token')

        self.mocker = unittest.mock.patch('dwave.experimental.fast_reverse_anneal.api.Client')
        self.mock_client = self.mocker.start()
        self.mock_client.from_config.return_value.__enter__.return_value = self.client

    def tearDown(self):
        self.client.close()
        self.mocker.stop()

        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def setup_mock_solvers(self, names):
        solvers = [
            Solver(client=None, data=structured_solver_data(name=name)) for name in names
        ]
        self.client._fetch_solvers = lambda **kw: solvers

    def test_internal_solver_selection(self):
        # prefer 'internal' when available
        self.setup_mock_solvers([
            "Advantage2_research2.3",
            "Advantage2_system4_x_internal",
            "Advantage2_research3",
        ])
        self.assertIn('internal', get_solver_name())

    def test_external_solver_selection(self):
        # select 'research' when internal not available
        self.setup_mock_solvers([
            "Advantage2_system4",
            "Advantage2_research2.3",
            "system3",
        ])
        self.assertIn('research', get_solver_name())

    def test_no_solver(self):
        # fail when solvers from `SOLVER_FILTER` not available
        self.setup_mock_solvers(["system1", "system2"])
        with self.assertRaises(SolverNotFoundError):
            get_solver_name()


class LiveSmokeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.sampler = DWaveSampler(solver=SOLVER_FILTER)
        except:
            raise unittest.SkipTest('Fast reverse annealing solver not available.')

    @classmethod
    def tearDownClass(cls):
        cls.sampler.close()

    def tearDown(self):
        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def test_get_parameters_from_sampler(self):
        params = get_parameters(self.sampler)
        self.assertIn('x_target_c', params)
        self.assertIn('x_nominal_pause_time', params)

    def test_get_parameters_from_name(self):
        params = get_parameters(get_solver_name())
        self.assertIn('x_target_c', params)
        self.assertIn('x_nominal_pause_time', params)

    def test_solver_selection(self):
        name = get_solver_name()
        self.assertIsNotNone(re.match(SOLVER_FILTER['name__regex'], name))
