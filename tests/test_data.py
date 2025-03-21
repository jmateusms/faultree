import numpy as np


fta1 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {
                    'id': '1',
                    'value': 0.1
                },
                {
                    'id': '2',
                    'value': 0.2
                }
            ]
        },
        {
            'id': '3',
            'value': 0.05
        }
    ]
}

rta1 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {
                    'id': '1',
                    'value': 0.9
                },
                {
                    'id': '2',
                    'value': 0.8
                }
            ]
        },
        {
            'id': '3',
            'value': 0.95
        }
    ]
}

fta1_result = 0.069
rta1_result = 0.931


fta2 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {
                    'id': '1',
                    'value': 0.2
                },
{
                    'id': '2',
                    'value': 0.1
                },
{
                    'id': '3',
                    'value': 0.5
                },
            ]
        },
        {
            'id': 'C',
            'gate': 'OR',
            'children': [
                {
                    'id': '4',
                    'value': 0.02
                },
{
                    'id': '5',
                    'value': 0.04
                }
            ]
        },
{
            'id': 'D',
            'gate': 'OR',
            'children': [
{
                    'id': 'E',
                    'gate': 'AND',
                    'children': [
                        {
                        'id': '7',
                        'value': 0.35
                        },
{
                        'id': '8',
                        'value': 0.2
                        },
{
                        'id': '9',
                        'value': 0.3
                        }
                    ]
                },
{
                    'id': '6',
                    'value': 0.01
                },
            ]
        }
    ]
}

rta2 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {
                    'id': '1',
                    'value': 0.8
                },
{
                    'id': '2',
                    'value': 0.9
                },
{
                    'id': '3',
                    'value': 0.5
                },
            ]
        },
        {
            'id': 'C',
            'gate': 'OR',
            'children': [
                {
                    'id': '4',
                    'value': 0.98
                },
{
                    'id': '5',
                    'value': 0.96
                }
            ]
        },
{
            'id': 'D',
            'gate': 'OR',
            'children': [
{
                    'id': 'E',
                    'gate': 'AND',
                    'children': [
                        {
                        'id': '7',
                        'value': 0.65
                        },
{
                        'id': '8',
                        'value': 0.8
                        },
{
                        'id': '9',
                        'value': 0.7
                        }
                    ]
                },
{
                    'id': '6',
                    'value': 0.99
                },
            ]
        }
    ]
}

fta2_result = 0.09728556
rta2_result = 0.90271444


fta3 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {

                    'id': '1',
                    'value': np.array([0.1, 0.2, 0.3])
                },
                {
                    'id': '2',
                    'value': np.array([0.15, 0.25, 0.35])
                }
            ]
        },
        {
            'id': '3',
            'value': np.array([0.01, 0.02, 0.05])
        }
    ]
}

rta3 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {
                    'id': '1',
                    'value': np.array([0.9, 0.8, 0.7])
                },
                {
                    'id': '2',
                    'value': np.array([0.85, 0.75, 0.65])
                }
            ]
        },
        {
            'id': '3',
            'value': np.array([0.99, 0.98, 0.95])
        }
    ]
}

fta3_result = np.array([0.02485, 0.069, 0.14975])
rta3_result = np.array([0.97515, 0.931, 0.85025])


fta4 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {
                    'id': '1',
                    'value': np.array([0.15, 0.2, 0.1, 0.25, 0.3])
                },
{
                    'id': '2',
                    'value': np.array([0.05, 0.15, 0.12, 0.18, 0.2])
                },
{
                    'id': '3',
                    'value': np.array([0.38, 0.45, 0.55, 0.37, 0.4])
                },
            ]
        },
        {
            'id': 'C',
            'gate': 'OR',
            'children': [
                {
                    'id': '4',
                    'value': np.array([0.01, 0.005, 0.008, 0.02, 0.03])
                },
{
                    'id': '5',
                    'value': np.array([0.05, 0.01, 0.03, 0.02, 0.04])
                }
            ]
        },
{
            'id': 'D',
            'gate': 'OR',
            'children': [
{
                    'id': 'E',
                    'gate': 'AND',
                    'children': [
                        {
                        'id': '7',
                        'value': np.array([0.3, 0.4, 0.32, 0.35, 0.38])
                        },
{
                        'id': '8',
                        'value': np.array([0.18, 0.22, 0.25, 0.19, 0.2])
                        },
{
                        'id': '9',
                        'value': np.array([0.3, 0.35, 0.39, 0.28, 0.27])
                        }
                    ]
                },
{
                    'id': '6',
                    'value': np.array([0.009, 0.01, 0.008, 0.012, 0.015])
                },
            ]
        }
    ]
}

rta4 = {
    'id': 'A',
    'gate': 'OR',
    'children': [
        {
            'id': 'B',
            'gate': 'AND',
            'children': [
                {
                    'id': '1',
                    'value': np.array([0.85, 0.8, 0.9, 0.75, 0.7])
                },
{
                    'id': '2',
                    'value': np.array([0.95, 0.85, 0.88, 0.82, 0.8])
                },
{
                    'id': '3',
                    'value': np.array([0.62, 0.55, 0.45, 0.63, 0.6])
                },
            ]
        },
        {
            'id': 'C',
            'gate': 'OR',
            'children': [
                {
                    'id': '4',
                    'value': np.array([0.99, 0.995, 0.992, 0.98, 0.97])
                },
{
                    'id': '5',
                    'value': np.array([0.95, 0.99, 0.97, 0.98, 0.96])
                }
            ]
        },
{
            'id': 'D',
            'gate': 'OR',
            'children': [
{
                    'id': 'E',
                    'gate': 'AND',
                    'children': [
                        {
                        'id': '7',
                        'value': np.array([0.7, 0.6, 0.68, 0.65, 0.62])
                        },
{
                        'id': '8',
                        'value': np.array([0.82, 0.78, 0.75, 0.81, 0.8])
                        },
{
                        'id': '9',
                        'value': np.array([0.7, 0.65, 0.61, 0.72, 0.73])
                        }
                    ]
                },
{
                    'id': '6',
                    'value': np.array([0.991, 0.99, 0.992, 0.988, 0.985])
                },
            ]
        }
    ]
}

fta4_result = np.array([0.085676744, 0.06759635, 0.081343051, 0.084297455, 0.12315145])
rta4_result = np.array([0.914323256, 0.93240365, 0.918656949, 0.915702545, 0.87684855])
