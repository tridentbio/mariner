import { Dataset } from '@app/rtk/generated/datasets'
import { readFileSync } from 'fs'
import path from 'path'

export const getDataset = () => {
  return {
    "name": "asdasd",
    "description": "asdasdasd",
    "rows": 4200,
    "columns": 3,
    "bytes": 282606,
    "stats": {
      "full": {
        "CMPD_CHEMBLID": {},
        "exp": {
          "hist": {
            "values": [
              {
                "bin_start": -1.5,
                "bin_end": -1.1,
                "count": 22
              },
              {
                "bin_start": -1.1,
                "bin_end": -0.7,
                "count": 68
              },
              {
                "bin_start": -0.7,
                "bin_end": -0.2999999999999998,
                "count": 75
              },
              {
                "bin_start": -0.2999999999999998,
                "bin_end": 0.10000000000000009,
                "count": 98
              },
              {
                "bin_start": 0.10000000000000009,
                "bin_end": 0.5,
                "count": 142
              },
              {
                "bin_start": 0.5,
                "bin_end": 0.9000000000000004,
                "count": 249
              },
              {
                "bin_start": 0.9000000000000004,
                "bin_end": 1.3000000000000003,
                "count": 312
              },
              {
                "bin_start": 1.3000000000000003,
                "bin_end": 1.7000000000000002,
                "count": 371
              },
              {
                "bin_start": 1.7000000000000002,
                "bin_end": 2.1,
                "count": 415
              },
              {
                "bin_start": 2.1,
                "bin_end": 2.5,
                "count": 525
              },
              {
                "bin_start": 2.5,
                "bin_end": 2.9000000000000004,
                "count": 628
              },
              {
                "bin_start": 2.9000000000000004,
                "bin_end": 3.3000000000000007,
                "count": 557
              },
              {
                "bin_start": 3.3000000000000007,
                "bin_end": 3.7,
                "count": 387
              },
              {
                "bin_start": 3.7,
                "bin_end": 4.1000000000000005,
                "count": 254
              },
              {
                "bin_start": 4.1000000000000005,
                "bin_end": 4.5,
                "count": 97
              },
              {
                "min": -1.5,
                "max": 4.5,
                "mean": 2.1863357142857143,
                "median": 2.36
              }
            ]
          }
        },
        "smiles": {
          "mwt": {
            "hist": {
              "values": [
                {
                  "bin_start": 113.084063972,
                  "bin_end": 213.52581746399994,
                  "count": 218
                },
                {
                  "bin_start": 213.52581746399994,
                  "bin_end": 313.96757095599986,
                  "count": 917
                },
                {
                  "bin_start": 313.96757095599986,
                  "bin_end": 414.4093244479998,
                  "count": 1378
                },
                {
                  "bin_start": 414.4093244479998,
                  "bin_end": 514.8510779399998,
                  "count": 1326
                },
                {
                  "bin_start": 514.8510779399998,
                  "bin_end": 615.2928314319996,
                  "count": 311
                },
                {
                  "bin_start": 615.2928314319996,
                  "bin_end": 715.7345849239996,
                  "count": 43
                },
                {
                  "bin_start": 715.7345849239996,
                  "bin_end": 816.1763384159995,
                  "count": 1
                },
                {
                  "bin_start": 816.1763384159995,
                  "bin_end": 916.6180919079994,
                  "count": 2
                },
                {
                  "bin_start": 916.6180919079994,
                  "bin_end": 1017.0598453999994,
                  "count": 1
                },
                {
                  "bin_start": 1017.0598453999994,
                  "bin_end": 1117.5015988919993,
                  "count": 1
                },
                {
                  "bin_start": 1117.5015988919993,
                  "bin_end": 1217.9433523839991,
                  "count": 0
                },
                {
                  "bin_start": 1217.9433523839991,
                  "bin_end": 1318.3851058759992,
                  "count": 0
                },
                {
                  "bin_start": 1318.3851058759992,
                  "bin_end": 1418.826859367999,
                  "count": 0
                },
                {
                  "bin_start": 1418.826859367999,
                  "bin_end": 1519.268612859999,
                  "count": 1
                },
                {
                  "bin_start": 1519.268612859999,
                  "bin_end": 1619.710366351999,
                  "count": 1
                },
                {
                  "min": 113.084063972,
                  "max": 1619.710366351999,
                  "mean": 382.6874463922821,
                  "median": 388.11100236000004
                }
              ]
            }
          },
          "tpsa": {
            "hist": {
              "values": [
                {
                  "bin_start": 0,
                  "bin_end": 46.80133333333333,
                  "count": 678
                },
                {
                  "bin_start": 46.80133333333333,
                  "bin_end": 93.60266666666666,
                  "count": 2220
                },
                {
                  "bin_start": 93.60266666666666,
                  "bin_end": 140.404,
                  "count": 1229
                },
                {
                  "bin_start": 140.404,
                  "bin_end": 187.20533333333333,
                  "count": 66
                },
                {
                  "bin_start": 187.20533333333333,
                  "bin_end": 234.00666666666666,
                  "count": 4
                },
                {
                  "bin_start": 234.00666666666666,
                  "bin_end": 280.808,
                  "count": 0
                },
                {
                  "bin_start": 280.808,
                  "bin_end": 327.6093333333333,
                  "count": 1
                },
                {
                  "bin_start": 327.6093333333333,
                  "bin_end": 374.41066666666666,
                  "count": 0
                },
                {
                  "bin_start": 374.41066666666666,
                  "bin_end": 421.212,
                  "count": 0
                },
                {
                  "bin_start": 421.212,
                  "bin_end": 468.0133333333333,
                  "count": 0
                },
                {
                  "bin_start": 468.0133333333333,
                  "bin_end": 514.8146666666667,
                  "count": 0
                },
                {
                  "bin_start": 514.8146666666667,
                  "bin_end": 561.616,
                  "count": 1
                },
                {
                  "bin_start": 561.616,
                  "bin_end": 608.4173333333333,
                  "count": 0
                },
                {
                  "bin_start": 608.4173333333333,
                  "bin_end": 655.2186666666666,
                  "count": 0
                },
                {
                  "bin_start": 655.2186666666666,
                  "bin_end": 702.02,
                  "count": 1
                },
                {
                  "min": 0,
                  "max": 702.02,
                  "mean": 78.84213095238096,
                  "median": 79.89999999999999
                }
              ]
            }
          },
          "atom_count": {
            "hist": {
              "values": [
                {
                  "bin_start": 7,
                  "bin_end": 14.2,
                  "count": 165
                },
                {
                  "bin_start": 14.2,
                  "bin_end": 21.4,
                  "count": 854
                },
                {
                  "bin_start": 21.4,
                  "bin_end": 28.6,
                  "count": 1327
                },
                {
                  "bin_start": 28.6,
                  "bin_end": 35.8,
                  "count": 1418
                },
                {
                  "bin_start": 35.8,
                  "bin_end": 43,
                  "count": 376
                },
                {
                  "bin_start": 43,
                  "bin_end": 50.2,
                  "count": 52
                },
                {
                  "bin_start": 50.2,
                  "bin_end": 57.4,
                  "count": 2
                },
                {
                  "bin_start": 57.4,
                  "bin_end": 64.6,
                  "count": 2
                },
                {
                  "bin_start": 64.6,
                  "bin_end": 71.8,
                  "count": 1
                },
                {
                  "bin_start": 71.8,
                  "bin_end": 79,
                  "count": 1
                },
                {
                  "bin_start": 79,
                  "bin_end": 86.2,
                  "count": 0
                },
                {
                  "bin_start": 86.2,
                  "bin_end": 93.4,
                  "count": 0
                },
                {
                  "bin_start": 93.4,
                  "bin_end": 100.60000000000001,
                  "count": 1
                },
                {
                  "bin_start": 100.60000000000001,
                  "bin_end": 107.8,
                  "count": 0
                },
                {
                  "bin_start": 107.8,
                  "bin_end": 115,
                  "count": 1
                },
                {
                  "min": 7,
                  "max": 115,
                  "mean": 27.04,
                  "median": 27
                }
              ]
            }
          },
          "ring_count": {
            "hist": {
              "values": [
                {
                  "bin_start": -0.5,
                  "bin_end": 0.5,
                  "count": 5
                },
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 113
                },
                {
                  "bin_start": 1.5,
                  "bin_end": 2.5,
                  "count": 630
                },
                {
                  "bin_start": 2.5,
                  "bin_end": 3.5,
                  "count": 1470
                },
                {
                  "bin_start": 3.5,
                  "bin_end": 4.5,
                  "count": 1263
                },
                {
                  "bin_start": 4.5,
                  "bin_end": 5.5,
                  "count": 554
                },
                {
                  "bin_start": 5.5,
                  "bin_end": 6.5,
                  "count": 130
                },
                {
                  "bin_start": 6.5,
                  "bin_end": 7.5,
                  "count": 28
                },
                {
                  "bin_start": 7.5,
                  "bin_end": 8.5,
                  "count": 6
                },
                {
                  "bin_start": 8.5,
                  "bin_end": 9.5,
                  "count": 0
                },
                {
                  "bin_start": 9.5,
                  "bin_end": 10.5,
                  "count": 0
                },
                {
                  "bin_start": 10.5,
                  "bin_end": 11.5,
                  "count": 0
                },
                {
                  "bin_start": 11.5,
                  "bin_end": 12.5,
                  "count": 0
                },
                {
                  "bin_start": 12.5,
                  "bin_end": 13.5,
                  "count": 1
                },
                {
                  "min": 0,
                  "max": 13,
                  "mean": 3.486190476190476,
                  "median": 3
                }
              ]
            }
          },
          "has_chiral_centers": {
            "hist": {
              "values": [
                {
                  "bin_start": -0.5,
                  "bin_end": 0.5,
                  "count": 3111
                },
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 1089
                },
                {
                  "min": 0,
                  "max": 1,
                  "mean": 0.2592857142857143,
                  "median": 0
                }
              ]
            }
          }
        },
        "step": {}
      },
      "train": {
        "CMPD_CHEMBLID": {},
        "exp": {
          "hist": {
            "values": [
              {
                "bin_start": -1.42,
                "bin_end": -1.0253333333333332,
                "count": 14
              },
              {
                "bin_start": -1.0253333333333332,
                "bin_end": -0.6306666666666666,
                "count": 38
              },
              {
                "bin_start": -0.6306666666666666,
                "bin_end": -0.236,
                "count": 43
              },
              {
                "bin_start": -0.236,
                "bin_end": 0.15866666666666673,
                "count": 54
              },
              {
                "bin_start": 0.15866666666666673,
                "bin_end": 0.5533333333333335,
                "count": 85
              },
              {
                "bin_start": 0.5533333333333335,
                "bin_end": 0.948,
                "count": 151
              },
              {
                "bin_start": 0.948,
                "bin_end": 1.3426666666666667,
                "count": 202
              },
              {
                "bin_start": 1.3426666666666667,
                "bin_end": 1.7373333333333334,
                "count": 229
              },
              {
                "bin_start": 1.7373333333333334,
                "bin_end": 2.132,
                "count": 266
              },
              {
                "bin_start": 2.132,
                "bin_end": 2.526666666666667,
                "count": 326
              },
              {
                "bin_start": 2.526666666666667,
                "bin_end": 2.921333333333333,
                "count": 326
              },
              {
                "bin_start": 2.921333333333333,
                "bin_end": 3.316,
                "count": 345
              },
              {
                "bin_start": 3.316,
                "bin_end": 3.7106666666666666,
                "count": 238
              },
              {
                "bin_start": 3.7106666666666666,
                "bin_end": 4.105333333333333,
                "count": 138
              },
              {
                "bin_start": 4.105333333333333,
                "bin_end": 4.5,
                "count": 65
              },
              {
                "min": -1.42,
                "max": 4.5,
                "mean": 2.2116547619047617,
                "median": 2.35
              }
            ]
          }
        },
        "smiles": {
          "mwt": {
            "hist": {
              "values": [
                {
                  "bin_start": 113.084063972,
                  "bin_end": 213.52581746399994,
                  "count": 120
                },
                {
                  "bin_start": 213.52581746399994,
                  "bin_end": 313.96757095599986,
                  "count": 540
                },
                {
                  "bin_start": 313.96757095599986,
                  "bin_end": 414.4093244479998,
                  "count": 832
                },
                {
                  "bin_start": 414.4093244479998,
                  "bin_end": 514.8510779399998,
                  "count": 805
                },
                {
                  "bin_start": 514.8510779399998,
                  "bin_end": 615.2928314319996,
                  "count": 194
                },
                {
                  "bin_start": 615.2928314319996,
                  "bin_end": 715.7345849239996,
                  "count": 24
                },
                {
                  "bin_start": 715.7345849239996,
                  "bin_end": 816.1763384159995,
                  "count": 1
                },
                {
                  "bin_start": 816.1763384159995,
                  "bin_end": 916.6180919079994,
                  "count": 1
                },
                {
                  "bin_start": 916.6180919079994,
                  "bin_end": 1017.0598453999994,
                  "count": 1
                },
                {
                  "bin_start": 1017.0598453999994,
                  "bin_end": 1117.5015988919993,
                  "count": 1
                },
                {
                  "bin_start": 1117.5015988919993,
                  "bin_end": 1217.9433523839991,
                  "count": 0
                },
                {
                  "bin_start": 1217.9433523839991,
                  "bin_end": 1318.3851058759992,
                  "count": 0
                },
                {
                  "bin_start": 1318.3851058759992,
                  "bin_end": 1418.826859367999,
                  "count": 0
                },
                {
                  "bin_start": 1418.826859367999,
                  "bin_end": 1519.268612859999,
                  "count": 0
                },
                {
                  "bin_start": 1519.268612859999,
                  "bin_end": 1619.710366351999,
                  "count": 1
                },
                {
                  "min": 113.084063972,
                  "max": 1619.710366351999,
                  "mean": 384.2943325779328,
                  "median": 388.17275012199997
                }
              ]
            }
          },
          "tpsa": {
            "hist": {
              "values": [
                {
                  "bin_start": 0,
                  "bin_end": 46.80133333333333,
                  "count": 399
                },
                {
                  "bin_start": 46.80133333333333,
                  "bin_end": 93.60266666666666,
                  "count": 1333
                },
                {
                  "bin_start": 93.60266666666666,
                  "bin_end": 140.404,
                  "count": 750
                },
                {
                  "bin_start": 140.404,
                  "bin_end": 187.20533333333333,
                  "count": 35
                },
                {
                  "bin_start": 187.20533333333333,
                  "bin_end": 234.00666666666666,
                  "count": 1
                },
                {
                  "bin_start": 234.00666666666666,
                  "bin_end": 280.808,
                  "count": 0
                },
                {
                  "bin_start": 280.808,
                  "bin_end": 327.6093333333333,
                  "count": 1
                },
                {
                  "bin_start": 327.6093333333333,
                  "bin_end": 374.41066666666666,
                  "count": 0
                },
                {
                  "bin_start": 374.41066666666666,
                  "bin_end": 421.212,
                  "count": 0
                },
                {
                  "bin_start": 421.212,
                  "bin_end": 468.0133333333333,
                  "count": 0
                },
                {
                  "bin_start": 468.0133333333333,
                  "bin_end": 514.8146666666667,
                  "count": 0
                },
                {
                  "bin_start": 514.8146666666667,
                  "bin_end": 561.616,
                  "count": 0
                },
                {
                  "bin_start": 561.616,
                  "bin_end": 608.4173333333333,
                  "count": 0
                },
                {
                  "bin_start": 608.4173333333333,
                  "bin_end": 655.2186666666666,
                  "count": 0
                },
                {
                  "bin_start": 655.2186666666666,
                  "bin_end": 702.02,
                  "count": 1
                },
                {
                  "min": 0,
                  "max": 702.02,
                  "mean": 79.13704365079366,
                  "median": 80.25999999999999
                }
              ]
            }
          },
          "atom_count": {
            "hist": {
              "values": [
                {
                  "bin_start": 7,
                  "bin_end": 14.2,
                  "count": 95
                },
                {
                  "bin_start": 14.2,
                  "bin_end": 21.4,
                  "count": 496
                },
                {
                  "bin_start": 21.4,
                  "bin_end": 28.6,
                  "count": 806
                },
                {
                  "bin_start": 28.6,
                  "bin_end": 35.8,
                  "count": 857
                },
                {
                  "bin_start": 35.8,
                  "bin_end": 43,
                  "count": 230
                },
                {
                  "bin_start": 43,
                  "bin_end": 50.2,
                  "count": 32
                },
                {
                  "bin_start": 50.2,
                  "bin_end": 57.4,
                  "count": 0
                },
                {
                  "bin_start": 57.4,
                  "bin_end": 64.6,
                  "count": 1
                },
                {
                  "bin_start": 64.6,
                  "bin_end": 71.8,
                  "count": 1
                },
                {
                  "bin_start": 71.8,
                  "bin_end": 79,
                  "count": 1
                },
                {
                  "bin_start": 79,
                  "bin_end": 86.2,
                  "count": 0
                },
                {
                  "bin_start": 86.2,
                  "bin_end": 93.4,
                  "count": 0
                },
                {
                  "bin_start": 93.4,
                  "bin_end": 100.60000000000001,
                  "count": 0
                },
                {
                  "bin_start": 100.60000000000001,
                  "bin_end": 107.8,
                  "count": 0
                },
                {
                  "bin_start": 107.8,
                  "bin_end": 115,
                  "count": 1
                },
                {
                  "min": 7,
                  "max": 115,
                  "mean": 27.1484126984127,
                  "median": 28
                }
              ]
            }
          },
          "ring_count": {
            "hist": {
              "values": [
                {
                  "bin_start": -0.5,
                  "bin_end": 0.5,
                  "count": 4
                },
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 65
                },
                {
                  "bin_start": 1.5,
                  "bin_end": 2.5,
                  "count": 360
                },
                {
                  "bin_start": 2.5,
                  "bin_end": 3.5,
                  "count": 863
                },
                {
                  "bin_start": 3.5,
                  "bin_end": 4.5,
                  "count": 781
                },
                {
                  "bin_start": 4.5,
                  "bin_end": 5.5,
                  "count": 348
                },
                {
                  "bin_start": 5.5,
                  "bin_end": 6.5,
                  "count": 81
                },
                {
                  "bin_start": 6.5,
                  "bin_end": 7.5,
                  "count": 15
                },
                {
                  "bin_start": 7.5,
                  "bin_end": 8.5,
                  "count": 3
                },
                {
                  "min": 0,
                  "max": 8,
                  "mean": 3.513095238095238,
                  "median": 3
                }
              ]
            }
          },
          "has_chiral_centers": {
            "hist": {
              "values": [
                {
                  "bin_start": -0.5,
                  "bin_end": 0.5,
                  "count": 1880
                },
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 640
                },
                {
                  "min": 0,
                  "max": 1,
                  "mean": 0.25396825396825395,
                  "median": 0
                }
              ]
            }
          }
        },
        "step": {}
      },
      "test": {
        "CMPD_CHEMBLID": {},
        "exp": {
          "hist": {
            "values": [
              {
                "bin_start": -1.5,
                "bin_end": -1.1,
                "count": 6
              },
              {
                "bin_start": -1.1,
                "bin_end": -0.7,
                "count": 18
              },
              {
                "bin_start": -0.7,
                "bin_end": -0.2999999999999998,
                "count": 16
              },
              {
                "bin_start": -0.2999999999999998,
                "bin_end": 0.10000000000000009,
                "count": 24
              },
              {
                "bin_start": 0.10000000000000009,
                "bin_end": 0.5,
                "count": 35
              },
              {
                "bin_start": 0.5,
                "bin_end": 0.9000000000000004,
                "count": 44
              },
              {
                "bin_start": 0.9000000000000004,
                "bin_end": 1.3000000000000003,
                "count": 56
              },
              {
                "bin_start": 1.3000000000000003,
                "bin_end": 1.7000000000000002,
                "count": 70
              },
              {
                "bin_start": 1.7000000000000002,
                "bin_end": 2.1,
                "count": 85
              },
              {
                "bin_start": 2.1,
                "bin_end": 2.5,
                "count": 101
              },
              {
                "bin_start": 2.5,
                "bin_end": 2.9000000000000004,
                "count": 141
              },
              {
                "bin_start": 2.9000000000000004,
                "bin_end": 3.3000000000000007,
                "count": 104
              },
              {
                "bin_start": 3.3000000000000007,
                "bin_end": 3.7,
                "count": 77
              },
              {
                "bin_start": 3.7,
                "bin_end": 4.1000000000000005,
                "count": 48
              },
              {
                "bin_start": 4.1000000000000005,
                "bin_end": 4.5,
                "count": 15
              },
              {
                "min": -1.5,
                "max": 4.5,
                "mean": 2.1386785714285717,
                "median": 2.37
              }
            ]
          }
        },
        "smiles": {
          "mwt": {
            "hist": {
              "values": [
                {
                  "bin_start": 118.053098192,
                  "bin_end": 155.3994789205333,
                  "count": 6
                },
                {
                  "bin_start": 155.3994789205333,
                  "bin_end": 192.74585964906663,
                  "count": 20
                },
                {
                  "bin_start": 192.74585964906663,
                  "bin_end": 230.09224037759998,
                  "count": 45
                },
                {
                  "bin_start": 230.09224037759998,
                  "bin_end": 267.43862110613327,
                  "count": 67
                },
                {
                  "bin_start": 267.43862110613327,
                  "bin_end": 304.7850018346666,
                  "count": 67
                },
                {
                  "bin_start": 304.7850018346666,
                  "bin_end": 342.13138256319996,
                  "count": 88
                },
                {
                  "bin_start": 342.13138256319996,
                  "bin_end": 379.47776329173325,
                  "count": 102
                },
                {
                  "bin_start": 379.47776329173325,
                  "bin_end": 416.8241440202666,
                  "count": 127
                },
                {
                  "bin_start": 416.8241440202666,
                  "bin_end": 454.17052474879995,
                  "count": 112
                },
                {
                  "bin_start": 454.17052474879995,
                  "bin_end": 491.51690547733324,
                  "count": 92
                },
                {
                  "bin_start": 491.51690547733324,
                  "bin_end": 528.8632862058666,
                  "count": 55
                },
                {
                  "bin_start": 528.8632862058666,
                  "bin_end": 566.2096669343999,
                  "count": 35
                },
                {
                  "bin_start": 566.2096669343999,
                  "bin_end": 603.5560476629332,
                  "count": 11
                },
                {
                  "bin_start": 603.5560476629332,
                  "bin_end": 640.9024283914665,
                  "count": 6
                },
                {
                  "bin_start": 640.9024283914665,
                  "bin_end": 678.2488091199999,
                  "count": 7
                },
                {
                  "min": 118.053098192,
                  "max": 678.2488091199999,
                  "mean": 379.98610512833096,
                  "median": 387.14921320400003
                }
              ]
            }
          },
          "tpsa": {
            "hist": {
              "values": [
                {
                  "bin_start": 3.24,
                  "bin_end": 16.58866666666667,
                  "count": 14
                },
                {
                  "bin_start": 16.58866666666667,
                  "bin_end": 29.937333333333335,
                  "count": 24
                },
                {
                  "bin_start": 29.937333333333335,
                  "bin_end": 43.286,
                  "count": 70
                },
                {
                  "bin_start": 43.286,
                  "bin_end": 56.63466666666667,
                  "count": 100
                },
                {
                  "bin_start": 56.63466666666667,
                  "bin_end": 69.98333333333333,
                  "count": 113
                },
                {
                  "bin_start": 69.98333333333333,
                  "bin_end": 83.332,
                  "count": 148
                },
                {
                  "bin_start": 83.332,
                  "bin_end": 96.68066666666665,
                  "count": 135
                },
                {
                  "bin_start": 96.68066666666665,
                  "bin_end": 110.02933333333333,
                  "count": 119
                },
                {
                  "bin_start": 110.02933333333333,
                  "bin_end": 123.378,
                  "count": 80
                },
                {
                  "bin_start": 123.378,
                  "bin_end": 136.7266666666667,
                  "count": 16
                },
                {
                  "bin_start": 136.7266666666667,
                  "bin_end": 150.07533333333333,
                  "count": 8
                },
                {
                  "bin_start": 150.07533333333333,
                  "bin_end": 163.424,
                  "count": 10
                },
                {
                  "bin_start": 163.424,
                  "bin_end": 176.77266666666668,
                  "count": 1
                },
                {
                  "bin_start": 176.77266666666668,
                  "bin_end": 190.12133333333333,
                  "count": 1
                },
                {
                  "bin_start": 190.12133333333333,
                  "bin_end": 203.47,
                  "count": 1
                },
                {
                  "min": 3.24,
                  "max": 203.47,
                  "mean": 78.43036904761905,
                  "median": 79.03
                }
              ]
            }
          },
          "atom_count": {
            "hist": {
              "values": [
                {
                  "bin_start": 9,
                  "bin_end": 11.8,
                  "count": 11
                },
                {
                  "bin_start": 11.8,
                  "bin_end": 14.6,
                  "count": 24
                },
                {
                  "bin_start": 14.6,
                  "bin_end": 17.4,
                  "count": 66
                },
                {
                  "bin_start": 17.4,
                  "bin_end": 20.2,
                  "count": 85
                },
                {
                  "bin_start": 20.2,
                  "bin_end": 23,
                  "count": 61
                },
                {
                  "bin_start": 23,
                  "bin_end": 25.799999999999997,
                  "count": 101
                },
                {
                  "bin_start": 25.799999999999997,
                  "bin_end": 28.599999999999998,
                  "count": 128
                },
                {
                  "bin_start": 28.599999999999998,
                  "bin_end": 31.4,
                  "count": 134
                },
                {
                  "bin_start": 31.4,
                  "bin_end": 34.2,
                  "count": 111
                },
                {
                  "bin_start": 34.2,
                  "bin_end": 37,
                  "count": 52
                },
                {
                  "bin_start": 37,
                  "bin_end": 39.8,
                  "count": 40
                },
                {
                  "bin_start": 39.8,
                  "bin_end": 42.599999999999994,
                  "count": 15
                },
                {
                  "bin_start": 42.599999999999994,
                  "bin_end": 45.4,
                  "count": 7
                },
                {
                  "bin_start": 45.4,
                  "bin_end": 48.199999999999996,
                  "count": 3
                },
                {
                  "bin_start": 48.199999999999996,
                  "bin_end": 51,
                  "count": 2
                },
                {
                  "min": 9,
                  "max": 51,
                  "mean": 26.789285714285715,
                  "median": 27
                }
              ]
            }
          },
          "ring_count": {
            "hist": {
              "values": [
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 17
                },
                {
                  "bin_start": 1.5,
                  "bin_end": 2.5,
                  "count": 140
                },
                {
                  "bin_start": 2.5,
                  "bin_end": 3.5,
                  "count": 294
                },
                {
                  "bin_start": 3.5,
                  "bin_end": 4.5,
                  "count": 250
                },
                {
                  "bin_start": 4.5,
                  "bin_end": 5.5,
                  "count": 106
                },
                {
                  "bin_start": 5.5,
                  "bin_end": 6.5,
                  "count": 26
                },
                {
                  "bin_start": 6.5,
                  "bin_end": 7.5,
                  "count": 4
                },
                {
                  "bin_start": 7.5,
                  "bin_end": 8.5,
                  "count": 2
                },
                {
                  "bin_start": 8.5,
                  "bin_end": 9.5,
                  "count": 0
                },
                {
                  "bin_start": 9.5,
                  "bin_end": 10.5,
                  "count": 0
                },
                {
                  "bin_start": 10.5,
                  "bin_end": 11.5,
                  "count": 0
                },
                {
                  "bin_start": 11.5,
                  "bin_end": 12.5,
                  "count": 0
                },
                {
                  "bin_start": 12.5,
                  "bin_end": 13.5,
                  "count": 1
                },
                {
                  "min": 1,
                  "max": 13,
                  "mean": 3.4785714285714286,
                  "median": 3
                }
              ]
            }
          },
          "has_chiral_centers": {
            "hist": {
              "values": [
                {
                  "bin_start": -0.5,
                  "bin_end": 0.5,
                  "count": 607
                },
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 233
                },
                {
                  "min": 0,
                  "max": 1,
                  "mean": 0.2773809523809524,
                  "median": 0
                }
              ]
            }
          }
        },
        "step": {}
      },
      "val": {
        "CMPD_CHEMBLID": {},
        "exp": {
          "hist": {
            "values": [
              {
                "bin_start": -1.45,
                "bin_end": -1.0533333333333332,
                "count": 8
              },
              {
                "bin_start": -1.0533333333333332,
                "bin_end": -0.6566666666666666,
                "count": 14
              },
              {
                "bin_start": -0.6566666666666666,
                "bin_end": -0.26,
                "count": 16
              },
              {
                "bin_start": -0.26,
                "bin_end": 0.13666666666666671,
                "count": 18
              },
              {
                "bin_start": 0.13666666666666671,
                "bin_end": 0.5333333333333334,
                "count": 41
              },
              {
                "bin_start": 0.5333333333333334,
                "bin_end": 0.9299999999999999,
                "count": 46
              },
              {
                "bin_start": 0.9299999999999999,
                "bin_end": 1.3266666666666669,
                "count": 60
              },
              {
                "bin_start": 1.3266666666666669,
                "bin_end": 1.7233333333333334,
                "count": 74
              },
              {
                "bin_start": 1.7233333333333334,
                "bin_end": 2.12,
                "count": 89
              },
              {
                "bin_start": 2.12,
                "bin_end": 2.5166666666666666,
                "count": 96
              },
              {
                "bin_start": 2.5166666666666666,
                "bin_end": 2.913333333333333,
                "count": 127
              },
              {
                "bin_start": 2.913333333333333,
                "bin_end": 3.3099999999999996,
                "count": 102
              },
              {
                "bin_start": 3.3099999999999996,
                "bin_end": 3.706666666666666,
                "count": 92
              },
              {
                "bin_start": 3.706666666666666,
                "bin_end": 4.1033333333333335,
                "count": 40
              },
              {
                "bin_start": 4.1033333333333335,
                "bin_end": 4.5,
                "count": 17
              },
              {
                "min": -1.45,
                "max": 4.5,
                "mean": 2.1580357142857145,
                "median": 2.36
              }
            ]
          }
        },
        "smiles": {
          "mwt": {
            "hist": {
              "values": [
                {
                  "bin_start": 115.02041815999999,
                  "bin_end": 202.13565303039996,
                  "count": 31
                },
                {
                  "bin_start": 202.13565303039996,
                  "bin_end": 289.25088790079997,
                  "count": 148
                },
                {
                  "bin_start": 289.25088790079997,
                  "bin_end": 376.3661227711999,
                  "count": 212
                },
                {
                  "bin_start": 376.3661227711999,
                  "bin_end": 463.48135764159986,
                  "count": 263
                },
                {
                  "bin_start": 463.48135764159986,
                  "bin_end": 550.5965925119998,
                  "count": 162
                },
                {
                  "bin_start": 550.5965925119998,
                  "bin_end": 637.7118273823999,
                  "count": 16
                },
                {
                  "bin_start": 637.7118273823999,
                  "bin_end": 724.8270622527998,
                  "count": 6
                },
                {
                  "bin_start": 724.8270622527998,
                  "bin_end": 811.9422971231998,
                  "count": 0
                },
                {
                  "bin_start": 811.9422971231998,
                  "bin_end": 899.0575319935997,
                  "count": 1
                },
                {
                  "bin_start": 899.0575319935997,
                  "bin_end": 986.1727668639996,
                  "count": 0
                },
                {
                  "bin_start": 986.1727668639996,
                  "bin_end": 1073.2880017343998,
                  "count": 0
                },
                {
                  "bin_start": 1073.2880017343998,
                  "bin_end": 1160.4032366047998,
                  "count": 0
                },
                {
                  "bin_start": 1160.4032366047998,
                  "bin_end": 1247.5184714751997,
                  "count": 0
                },
                {
                  "bin_start": 1247.5184714751997,
                  "bin_end": 1334.6337063455996,
                  "count": 0
                },
                {
                  "bin_start": 1334.6337063455996,
                  "bin_end": 1421.7489412159996,
                  "count": 1
                },
                {
                  "min": 115.02041815999999,
                  "max": 1421.7489412159996,
                  "mean": 380.56812909928095,
                  "median": 386.12273268
                }
              ]
            }
          },
          "tpsa": {
            "hist": {
              "values": [
                {
                  "bin_start": 3.24,
                  "bin_end": 38.41533333333332,
                  "count": 79
                },
                {
                  "bin_start": 38.41533333333332,
                  "bin_end": 73.59066666666664,
                  "count": 287
                },
                {
                  "bin_start": 73.59066666666664,
                  "bin_end": 108.76599999999995,
                  "count": 361
                },
                {
                  "bin_start": 108.76599999999995,
                  "bin_end": 143.9413333333333,
                  "count": 99
                },
                {
                  "bin_start": 143.9413333333333,
                  "bin_end": 179.11666666666662,
                  "count": 10
                },
                {
                  "bin_start": 179.11666666666662,
                  "bin_end": 214.29199999999992,
                  "count": 2
                },
                {
                  "bin_start": 214.29199999999992,
                  "bin_end": 249.46733333333324,
                  "count": 1
                },
                {
                  "bin_start": 249.46733333333324,
                  "bin_end": 284.64266666666657,
                  "count": 0
                },
                {
                  "bin_start": 284.64266666666657,
                  "bin_end": 319.81799999999987,
                  "count": 0
                },
                {
                  "bin_start": 319.81799999999987,
                  "bin_end": 354.9933333333332,
                  "count": 0
                },
                {
                  "bin_start": 354.9933333333332,
                  "bin_end": 390.1686666666665,
                  "count": 0
                },
                {
                  "bin_start": 390.1686666666665,
                  "bin_end": 425.3439999999998,
                  "count": 0
                },
                {
                  "bin_start": 425.3439999999998,
                  "bin_end": 460.5193333333332,
                  "count": 0
                },
                {
                  "bin_start": 460.5193333333332,
                  "bin_end": 495.6946666666665,
                  "count": 0
                },
                {
                  "bin_start": 495.6946666666665,
                  "bin_end": 530.8699999999998,
                  "count": 1
                },
                {
                  "min": 3.24,
                  "max": 530.8699999999998,
                  "mean": 78.36915476190475,
                  "median": 79.155
                }
              ]
            }
          },
          "atom_count": {
            "hist": {
              "values": [
                {
                  "bin_start": 7,
                  "bin_end": 13.2,
                  "count": 29
                },
                {
                  "bin_start": 13.2,
                  "bin_end": 19.4,
                  "count": 117
                },
                {
                  "bin_start": 19.4,
                  "bin_end": 25.6,
                  "count": 207
                },
                {
                  "bin_start": 25.6,
                  "bin_end": 31.8,
                  "count": 237
                },
                {
                  "bin_start": 31.8,
                  "bin_end": 38,
                  "count": 207
                },
                {
                  "bin_start": 38,
                  "bin_end": 44.2,
                  "count": 35
                },
                {
                  "bin_start": 44.2,
                  "bin_end": 50.4,
                  "count": 5
                },
                {
                  "bin_start": 50.4,
                  "bin_end": 56.6,
                  "count": 1
                },
                {
                  "bin_start": 56.6,
                  "bin_end": 62.800000000000004,
                  "count": 1
                },
                {
                  "bin_start": 62.800000000000004,
                  "bin_end": 69,
                  "count": 0
                },
                {
                  "bin_start": 69,
                  "bin_end": 75.2,
                  "count": 0
                },
                {
                  "bin_start": 75.2,
                  "bin_end": 81.4,
                  "count": 0
                },
                {
                  "bin_start": 81.4,
                  "bin_end": 87.60000000000001,
                  "count": 0
                },
                {
                  "bin_start": 87.60000000000001,
                  "bin_end": 93.8,
                  "count": 0
                },
                {
                  "bin_start": 93.8,
                  "bin_end": 100,
                  "count": 1
                },
                {
                  "min": 7,
                  "max": 100,
                  "mean": 26.965476190476192,
                  "median": 27
                }
              ]
            }
          },
          "ring_count": {
            "hist": {
              "values": [
                {
                  "bin_start": -0.5,
                  "bin_end": 0.5,
                  "count": 1
                },
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 31
                },
                {
                  "bin_start": 1.5,
                  "bin_end": 2.5,
                  "count": 130
                },
                {
                  "bin_start": 2.5,
                  "bin_end": 3.5,
                  "count": 313
                },
                {
                  "bin_start": 3.5,
                  "bin_end": 4.5,
                  "count": 232
                },
                {
                  "bin_start": 4.5,
                  "bin_end": 5.5,
                  "count": 100
                },
                {
                  "bin_start": 5.5,
                  "bin_end": 6.5,
                  "count": 23
                },
                {
                  "bin_start": 6.5,
                  "bin_end": 7.5,
                  "count": 9
                },
                {
                  "bin_start": 7.5,
                  "bin_end": 8.5,
                  "count": 1
                },
                {
                  "min": 0,
                  "max": 8,
                  "mean": 3.413095238095238,
                  "median": 3
                }
              ]
            }
          },
          "has_chiral_centers": {
            "hist": {
              "values": [
                {
                  "bin_start": -0.5,
                  "bin_end": 0.5,
                  "count": 624
                },
                {
                  "bin_start": 0.5,
                  "bin_end": 1.5,
                  "count": 216
                },
                {
                  "min": 0,
                  "max": 1,
                  "mean": 0.2571428571428571,
                  "median": 0
                }
              ]
            }
          }
        },
        "step": {}
      }
    },
    "dataUrl": "datasets/d8346fdfebeced6c538392764cc3f298.csv",
    "splitTarget": "60-20-20",
    "splitActual": null,
    "splitType": "random",
    "createdAt": "2022-12-23T14:21:02.561594+00:00",
    "updatedAt": "2022-12-23T14:21:02.561597+00:00",
    "createdById": 66,
    "columnsMetadata": [
      {
        "dataType": {
          "domainKind": "string"
        },
        "description": "",
        "pattern": "CMPD_CHEMBLID",
        "datasetId": 155
      },
      {
        "dataType": {
          "domainKind": "numeric",
          "unit": "mole"
        },
        "description": "",
        "pattern": "exp",
        "datasetId": 155
      },
      {
        "dataType": {
          "domainKind": "smiles"
        },
        "description": "asdasdas",
        "pattern": "smiles",
        "datasetId": 155
      }
    ],
    "id": 155
  }
}
