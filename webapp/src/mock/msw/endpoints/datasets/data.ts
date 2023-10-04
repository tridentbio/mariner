export const zincCsvMetadata = [
  {
    name: "zinc_id",
    dtype: {
      domainKind: "string"
    }
  },
  {
    name: "smiles",
    dtype: {
      domainKind: "smiles"
    }
  },
  {
    name: "mwt",
    dtype: {
      domainKind: "numeric"
    }
  },
  {
    name: "tpsa",
    dtype: {
      domainKind: "numeric"
    }
  },
  {
    name: "mwt_group",
    dtype: {
      domainKind: "categorical",
      classes: {
        mwt_big: 0,
        mwt_small: 1
      }
    }
  }
]


export const datasetsData = [
  {
    id: 1,
    name: 'IRIS_DATASET_NAME',
    description: 'rhuxbedsfawiyrddnoicssmc',
    rows: 10,
    columns: 8,
    bytes: 340,
    stats: {
      full: {
        sepal_length: {
          hist: {
            values: [
              {
                bin_start: 4.8,
                bin_end: 4.993333333333333,
                count: 2,
              },
              {
                bin_start: 4.993333333333333,
                bin_end: 5.1866666666666665,
                count: 0,
              },
              {
                bin_start: 5.1866666666666665,
                bin_end: 5.38,
                count: 1,
              },
              {
                bin_start: 5.38,
                bin_end: 5.573333333333333,
                count: 0,
              },
              {
                bin_start: 5.573333333333333,
                bin_end: 5.766666666666667,
                count: 0,
              },
              {
                bin_start: 5.766666666666667,
                bin_end: 5.96,
                count: 2,
              },
              {
                bin_start: 5.96,
                bin_end: 6.153333333333333,
                count: 0,
              },
              {
                bin_start: 6.153333333333333,
                bin_end: 6.346666666666667,
                count: 1,
              },
              {
                bin_start: 6.346666666666667,
                bin_end: 6.54,
                count: 1,
              },
              {
                bin_start: 6.54,
                bin_end: 6.733333333333333,
                count: 0,
              },
              {
                bin_start: 6.733333333333333,
                bin_end: 6.926666666666667,
                count: 2,
              },
              {
                bin_start: 6.926666666666667,
                bin_end: 7.12,
                count: 0,
              },
              {
                bin_start: 7.12,
                bin_end: 7.3133333333333335,
                count: 0,
              },
              {
                bin_start: 7.3133333333333335,
                bin_end: 7.506666666666667,
                count: 0,
              },
              {
                bin_start: 7.506666666666667,
                bin_end: 7.7,
                count: 1,
              },
              {
                min: 4.8,
                max: 7.7,
                mean: 6.06,
                median: 6.05,
              },
            ],
          },
        },
        sepal_width: {
          hist: {
            values: [
              {
                bin_start: 2.6,
                bin_end: 2.68,
                count: 1,
              },
              {
                bin_start: 2.68,
                bin_end: 2.7600000000000002,
                count: 1,
              },
              {
                bin_start: 2.7600000000000002,
                bin_end: 2.84,
                count: 1,
              },
              {
                bin_start: 2.84,
                bin_end: 2.92,
                count: 1,
              },
              {
                bin_start: 2.92,
                bin_end: 3,
                count: 0,
              },
              {
                bin_start: 3,
                bin_end: 3.08,
                count: 1,
              },
              {
                bin_start: 3.08,
                bin_end: 3.16,
                count: 1,
              },
              {
                bin_start: 3.16,
                bin_end: 3.24,
                count: 0,
              },
              {
                bin_start: 3.24,
                bin_end: 3.32,
                count: 0,
              },
              {
                bin_start: 3.32,
                bin_end: 3.4,
                count: 0,
              },
              {
                bin_start: 3.4,
                bin_end: 3.48,
                count: 2,
              },
              {
                bin_start: 3.48,
                bin_end: 3.56,
                count: 1,
              },
              {
                bin_start: 3.56,
                bin_end: 3.6399999999999997,
                count: 0,
              },
              {
                bin_start: 3.6399999999999997,
                bin_end: 3.7199999999999998,
                count: 0,
              },
              {
                bin_start: 3.7199999999999998,
                bin_end: 3.8,
                count: 1,
              },
              {
                min: 2.6,
                max: 3.8,
                mean: 3.12,
                median: 3.05,
              },
            ],
          },
        },
        petal_length: {
          hist: {
            values: [
              {
                bin_start: 1.5,
                bin_end: 1.8466666666666667,
                count: 2,
              },
              {
                bin_start: 1.8466666666666667,
                bin_end: 2.1933333333333334,
                count: 1,
              },
              {
                bin_start: 2.1933333333333334,
                bin_end: 2.54,
                count: 0,
              },
              {
                bin_start: 2.54,
                bin_end: 2.8866666666666667,
                count: 0,
              },
              {
                bin_start: 2.8866666666666667,
                bin_end: 3.2333333333333334,
                count: 0,
              },
              {
                bin_start: 3.2333333333333334,
                bin_end: 3.58,
                count: 0,
              },
              {
                bin_start: 3.58,
                bin_end: 3.9266666666666667,
                count: 0,
              },
              {
                bin_start: 3.9266666666666667,
                bin_end: 4.273333333333333,
                count: 2,
              },
              {
                bin_start: 4.273333333333333,
                bin_end: 4.62,
                count: 0,
              },
              {
                bin_start: 4.62,
                bin_end: 4.966666666666667,
                count: 2,
              },
              {
                bin_start: 4.966666666666667,
                bin_end: 5.3133333333333335,
                count: 0,
              },
              {
                bin_start: 5.3133333333333335,
                bin_end: 5.66,
                count: 2,
              },
              {
                bin_start: 5.66,
                bin_end: 6.006666666666667,
                count: 0,
              },
              {
                bin_start: 6.006666666666667,
                bin_end: 6.3533333333333335,
                count: 0,
              },
              {
                bin_start: 6.3533333333333335,
                bin_end: 6.7,
                count: 1,
              },
              {
                min: 1.5,
                max: 6.7,
                mean: 4.0600000000000005,
                median: 4.449999999999999,
              },
            ],
          },
        },
        petal_width: {
          hist: {
            values: [
              {
                bin_start: 0.2,
                bin_end: 0.33333333333333337,
                count: 3,
              },
              {
                bin_start: 0.33333333333333337,
                bin_end: 0.4666666666666667,
                count: 0,
              },
              {
                bin_start: 0.4666666666666667,
                bin_end: 0.6000000000000001,
                count: 0,
              },
              {
                bin_start: 0.6000000000000001,
                bin_end: 0.7333333333333334,
                count: 0,
              },
              {
                bin_start: 0.7333333333333334,
                bin_end: 0.8666666666666667,
                count: 0,
              },
              {
                bin_start: 0.8666666666666667,
                bin_end: 1,
                count: 0,
              },
              {
                bin_start: 1,
                bin_end: 1.1333333333333333,
                count: 1,
              },
              {
                bin_start: 1.1333333333333333,
                bin_end: 1.2666666666666666,
                count: 1,
              },
              {
                bin_start: 1.2666666666666666,
                bin_end: 1.4,
                count: 0,
              },
              {
                bin_start: 1.4,
                bin_end: 1.5333333333333332,
                count: 2,
              },
              {
                bin_start: 1.5333333333333332,
                bin_end: 1.6666666666666665,
                count: 0,
              },
              {
                bin_start: 1.6666666666666665,
                bin_end: 1.8,
                count: 0,
              },
              {
                bin_start: 1.8,
                bin_end: 1.9333333333333333,
                count: 2,
              },
              {
                bin_start: 1.9333333333333333,
                bin_end: 2.066666666666667,
                count: 0,
              },
              {
                bin_start: 2.066666666666667,
                bin_end: 2.2,
                count: 1,
              },
              {
                min: 0.2,
                max: 2.2,
                mean: 1.15,
                median: 1.2999999999999998,
              },
            ],
          },
        },
        species: {
          hist: {
            values: [
              {
                label: '1',
                count: 4,
              },
              {
                label: '2',
                count: 3,
              },
              {
                label: '0',
                count: 3,
              },
            ],
          },
        },
        large_petal_length: {
          hist: {
            values: [
              {
                label: '0',
                count: 7,
              },
              {
                label: '1',
                count: 3,
              },
            ],
          },
        },
        large_petal_width: {
          hist: {
            values: [
              {
                label: '0',
                count: 7,
              },
              {
                label: '1',
                count: 3,
              },
            ],
          },
        },
        step: {},
      },
      train: {
        sepal_length: {
          hist: {
            values: [
              {
                bin_start: 4.8,
                bin_end: 4.913333333333333,
                count: 2,
              },
              {
                bin_start: 4.913333333333333,
                bin_end: 5.026666666666666,
                count: 0,
              },
              {
                bin_start: 5.026666666666666,
                bin_end: 5.14,
                count: 0,
              },
              {
                bin_start: 5.14,
                bin_end: 5.253333333333333,
                count: 0,
              },
              {
                bin_start: 5.253333333333333,
                bin_end: 5.366666666666666,
                count: 0,
              },
              {
                bin_start: 5.366666666666666,
                bin_end: 5.4799999999999995,
                count: 0,
              },
              {
                bin_start: 5.4799999999999995,
                bin_end: 5.593333333333334,
                count: 0,
              },
              {
                bin_start: 5.593333333333334,
                bin_end: 5.706666666666667,
                count: 0,
              },
              {
                bin_start: 5.706666666666667,
                bin_end: 5.82,
                count: 2,
              },
              {
                bin_start: 5.82,
                bin_end: 5.933333333333334,
                count: 0,
              },
              {
                bin_start: 5.933333333333334,
                bin_end: 6.046666666666667,
                count: 0,
              },
              {
                bin_start: 6.046666666666667,
                bin_end: 6.16,
                count: 0,
              },
              {
                bin_start: 6.16,
                bin_end: 6.273333333333333,
                count: 0,
              },
              {
                bin_start: 6.273333333333333,
                bin_end: 6.386666666666667,
                count: 1,
              },
              {
                bin_start: 6.386666666666667,
                bin_end: 6.5,
                count: 1,
              },
              {
                min: 4.8,
                max: 6.5,
                mean: 5.666666666666667,
                median: 5.8,
              },
            ],
          },
        },
        sepal_width: {
          hist: {
            values: [
              {
                bin_start: 2.6,
                bin_end: 2.6533333333333333,
                count: 1,
              },
              {
                bin_start: 2.6533333333333333,
                bin_end: 2.7066666666666666,
                count: 1,
              },
              {
                bin_start: 2.7066666666666666,
                bin_end: 2.7600000000000002,
                count: 0,
              },
              {
                bin_start: 2.7600000000000002,
                bin_end: 2.8133333333333335,
                count: 0,
              },
              {
                bin_start: 2.8133333333333335,
                bin_end: 2.8666666666666667,
                count: 0,
              },
              {
                bin_start: 2.8666666666666667,
                bin_end: 2.92,
                count: 1,
              },
              {
                bin_start: 2.92,
                bin_end: 2.973333333333333,
                count: 0,
              },
              {
                bin_start: 2.973333333333333,
                bin_end: 3.026666666666667,
                count: 1,
              },
              {
                bin_start: 3.026666666666667,
                bin_end: 3.08,
                count: 0,
              },
              {
                bin_start: 3.08,
                bin_end: 3.1333333333333333,
                count: 0,
              },
              {
                bin_start: 3.1333333333333333,
                bin_end: 3.1866666666666665,
                count: 0,
              },
              {
                bin_start: 3.1866666666666665,
                bin_end: 3.24,
                count: 0,
              },
              {
                bin_start: 3.24,
                bin_end: 3.2933333333333334,
                count: 0,
              },
              {
                bin_start: 3.2933333333333334,
                bin_end: 3.3466666666666667,
                count: 0,
              },
              {
                bin_start: 3.3466666666666667,
                bin_end: 3.4,
                count: 2,
              },
              {
                min: 2.6,
                max: 3.4,
                mean: 3,
                median: 2.95,
              },
            ],
          },
        },
        petal_length: {
          hist: {
            values: [
              {
                bin_start: 1.6,
                bin_end: 1.8666666666666667,
                count: 1,
              },
              {
                bin_start: 1.8666666666666667,
                bin_end: 2.1333333333333333,
                count: 1,
              },
              {
                bin_start: 2.1333333333333333,
                bin_end: 2.4000000000000004,
                count: 0,
              },
              {
                bin_start: 2.4000000000000004,
                bin_end: 2.666666666666667,
                count: 0,
              },
              {
                bin_start: 2.666666666666667,
                bin_end: 2.9333333333333336,
                count: 0,
              },
              {
                bin_start: 2.9333333333333336,
                bin_end: 3.2,
                count: 0,
              },
              {
                bin_start: 3.2,
                bin_end: 3.466666666666667,
                count: 0,
              },
              {
                bin_start: 3.466666666666667,
                bin_end: 3.7333333333333334,
                count: 0,
              },
              {
                bin_start: 3.7333333333333334,
                bin_end: 4,
                count: 0,
              },
              {
                bin_start: 4,
                bin_end: 4.266666666666667,
                count: 2,
              },
              {
                bin_start: 4.266666666666667,
                bin_end: 4.533333333333333,
                count: 0,
              },
              {
                bin_start: 4.533333333333333,
                bin_end: 4.800000000000001,
                count: 0,
              },
              {
                bin_start: 4.800000000000001,
                bin_end: 5.066666666666666,
                count: 0,
              },
              {
                bin_start: 5.066666666666666,
                bin_end: 5.333333333333334,
                count: 0,
              },
              {
                bin_start: 5.333333333333334,
                bin_end: 5.6,
                count: 2,
              },
              {
                min: 1.6,
                max: 5.6,
                mean: 3.7833333333333337,
                median: 4.05,
              },
            ],
          },
        },
        petal_width: {
          hist: {
            values: [
              {
                bin_start: 0.2,
                bin_end: 0.3066666666666667,
                count: 2,
              },
              {
                bin_start: 0.3066666666666667,
                bin_end: 0.41333333333333333,
                count: 0,
              },
              {
                bin_start: 0.41333333333333333,
                bin_end: 0.52,
                count: 0,
              },
              {
                bin_start: 0.52,
                bin_end: 0.6266666666666667,
                count: 0,
              },
              {
                bin_start: 0.6266666666666667,
                bin_end: 0.7333333333333334,
                count: 0,
              },
              {
                bin_start: 0.7333333333333334,
                bin_end: 0.8400000000000001,
                count: 0,
              },
              {
                bin_start: 0.8400000000000001,
                bin_end: 0.9466666666666668,
                count: 0,
              },
              {
                bin_start: 0.9466666666666668,
                bin_end: 1.0533333333333335,
                count: 1,
              },
              {
                bin_start: 1.0533333333333335,
                bin_end: 1.1600000000000001,
                count: 0,
              },
              {
                bin_start: 1.1600000000000001,
                bin_end: 1.2666666666666666,
                count: 1,
              },
              {
                bin_start: 1.2666666666666666,
                bin_end: 1.3733333333333333,
                count: 0,
              },
              {
                bin_start: 1.3733333333333333,
                bin_end: 1.48,
                count: 0,
              },
              {
                bin_start: 1.48,
                bin_end: 1.5866666666666667,
                count: 0,
              },
              {
                bin_start: 1.5866666666666667,
                bin_end: 1.6933333333333334,
                count: 0,
              },
              {
                bin_start: 1.6933333333333334,
                bin_end: 1.8,
                count: 2,
              },
              {
                min: 0.2,
                max: 1.8,
                mean: 1.0333333333333334,
                median: 1.1,
              },
            ],
          },
        },
        species: {
          hist: {
            values: [
              {
                label: '2',
                count: 2,
              },
              {
                label: '1',
                count: 2,
              },
              {
                label: '0',
                count: 2,
              },
            ],
          },
        },
        large_petal_length: {
          hist: {
            values: [
              {
                label: '0',
                count: 4,
              },
              {
                label: '1',
                count: 2,
              },
            ],
          },
        },
        large_petal_width: {
          hist: {
            values: [
              {
                label: '0',
                count: 4,
              },
              {
                label: '1',
                count: 2,
              },
            ],
          },
        },
        step: {},
      },
      test: {
        sepal_length: {
          hist: {
            values: [
              {
                bin_start: 5.2,
                bin_end: 5.3133333333333335,
                count: 1,
              },
              {
                bin_start: 5.3133333333333335,
                bin_end: 5.426666666666667,
                count: 0,
              },
              {
                bin_start: 5.426666666666667,
                bin_end: 5.54,
                count: 0,
              },
              {
                bin_start: 5.54,
                bin_end: 5.653333333333333,
                count: 0,
              },
              {
                bin_start: 5.653333333333333,
                bin_end: 5.766666666666667,
                count: 0,
              },
              {
                bin_start: 5.766666666666667,
                bin_end: 5.88,
                count: 0,
              },
              {
                bin_start: 5.88,
                bin_end: 5.993333333333334,
                count: 0,
              },
              {
                bin_start: 5.993333333333334,
                bin_end: 6.106666666666667,
                count: 0,
              },
              {
                bin_start: 6.106666666666667,
                bin_end: 6.220000000000001,
                count: 0,
              },
              {
                bin_start: 6.220000000000001,
                bin_end: 6.333333333333334,
                count: 0,
              },
              {
                bin_start: 6.333333333333334,
                bin_end: 6.446666666666667,
                count: 0,
              },
              {
                bin_start: 6.446666666666667,
                bin_end: 6.5600000000000005,
                count: 0,
              },
              {
                bin_start: 6.5600000000000005,
                bin_end: 6.673333333333334,
                count: 0,
              },
              {
                bin_start: 6.673333333333334,
                bin_end: 6.786666666666667,
                count: 0,
              },
              {
                bin_start: 6.786666666666667,
                bin_end: 6.9,
                count: 1,
              },
              {
                min: 5.2,
                max: 6.9,
                mean: 6.050000000000001,
                median: 6.050000000000001,
              },
            ],
          },
        },
        sepal_width: {
          hist: {
            values: [
              {
                bin_start: 3.1,
                bin_end: 3.126666666666667,
                count: 1,
              },
              {
                bin_start: 3.126666666666667,
                bin_end: 3.1533333333333333,
                count: 0,
              },
              {
                bin_start: 3.1533333333333333,
                bin_end: 3.18,
                count: 0,
              },
              {
                bin_start: 3.18,
                bin_end: 3.2066666666666666,
                count: 0,
              },
              {
                bin_start: 3.2066666666666666,
                bin_end: 3.2333333333333334,
                count: 0,
              },
              {
                bin_start: 3.2333333333333334,
                bin_end: 3.2600000000000002,
                count: 0,
              },
              {
                bin_start: 3.2600000000000002,
                bin_end: 3.2866666666666666,
                count: 0,
              },
              {
                bin_start: 3.2866666666666666,
                bin_end: 3.3133333333333335,
                count: 0,
              },
              {
                bin_start: 3.3133333333333335,
                bin_end: 3.34,
                count: 0,
              },
              {
                bin_start: 3.34,
                bin_end: 3.3666666666666667,
                count: 0,
              },
              {
                bin_start: 3.3666666666666667,
                bin_end: 3.3933333333333335,
                count: 0,
              },
              {
                bin_start: 3.3933333333333335,
                bin_end: 3.42,
                count: 0,
              },
              {
                bin_start: 3.42,
                bin_end: 3.4466666666666668,
                count: 0,
              },
              {
                bin_start: 3.4466666666666668,
                bin_end: 3.473333333333333,
                count: 0,
              },
              {
                bin_start: 3.473333333333333,
                bin_end: 3.5,
                count: 1,
              },
              {
                min: 3.1,
                max: 3.5,
                mean: 3.3,
                median: 3.3,
              },
            ],
          },
        },
        petal_length: {
          hist: {
            values: [
              {
                bin_start: 1.5,
                bin_end: 1.7266666666666666,
                count: 1,
              },
              {
                bin_start: 1.7266666666666666,
                bin_end: 1.9533333333333334,
                count: 0,
              },
              {
                bin_start: 1.9533333333333334,
                bin_end: 2.18,
                count: 0,
              },
              {
                bin_start: 2.18,
                bin_end: 2.4066666666666667,
                count: 0,
              },
              {
                bin_start: 2.4066666666666667,
                bin_end: 2.6333333333333333,
                count: 0,
              },
              {
                bin_start: 2.6333333333333333,
                bin_end: 2.8600000000000003,
                count: 0,
              },
              {
                bin_start: 2.8600000000000003,
                bin_end: 3.086666666666667,
                count: 0,
              },
              {
                bin_start: 3.086666666666667,
                bin_end: 3.3133333333333335,
                count: 0,
              },
              {
                bin_start: 3.3133333333333335,
                bin_end: 3.54,
                count: 0,
              },
              {
                bin_start: 3.54,
                bin_end: 3.7666666666666666,
                count: 0,
              },
              {
                bin_start: 3.7666666666666666,
                bin_end: 3.9933333333333336,
                count: 0,
              },
              {
                bin_start: 3.9933333333333336,
                bin_end: 4.220000000000001,
                count: 0,
              },
              {
                bin_start: 4.220000000000001,
                bin_end: 4.446666666666667,
                count: 0,
              },
              {
                bin_start: 4.446666666666667,
                bin_end: 4.673333333333334,
                count: 0,
              },
              {
                bin_start: 4.673333333333334,
                bin_end: 4.9,
                count: 1,
              },
              {
                min: 1.5,
                max: 4.9,
                mean: 3.2,
                median: 3.2,
              },
            ],
          },
        },
        petal_width: {
          hist: {
            values: [
              {
                bin_start: 0.2,
                bin_end: 0.2866666666666667,
                count: 1,
              },
              {
                bin_start: 0.2866666666666667,
                bin_end: 0.37333333333333335,
                count: 0,
              },
              {
                bin_start: 0.37333333333333335,
                bin_end: 0.46,
                count: 0,
              },
              {
                bin_start: 0.46,
                bin_end: 0.5466666666666666,
                count: 0,
              },
              {
                bin_start: 0.5466666666666666,
                bin_end: 0.6333333333333333,
                count: 0,
              },
              {
                bin_start: 0.6333333333333333,
                bin_end: 0.72,
                count: 0,
              },
              {
                bin_start: 0.72,
                bin_end: 0.8066666666666666,
                count: 0,
              },
              {
                bin_start: 0.8066666666666666,
                bin_end: 0.8933333333333333,
                count: 0,
              },
              {
                bin_start: 0.8933333333333333,
                bin_end: 0.98,
                count: 0,
              },
              {
                bin_start: 0.98,
                bin_end: 1.0666666666666667,
                count: 0,
              },
              {
                bin_start: 1.0666666666666667,
                bin_end: 1.1533333333333333,
                count: 0,
              },
              {
                bin_start: 1.1533333333333333,
                bin_end: 1.24,
                count: 0,
              },
              {
                bin_start: 1.24,
                bin_end: 1.3266666666666667,
                count: 0,
              },
              {
                bin_start: 1.3266666666666667,
                bin_end: 1.4133333333333333,
                count: 0,
              },
              {
                bin_start: 1.4133333333333333,
                bin_end: 1.5,
                count: 1,
              },
              {
                min: 0.2,
                max: 1.5,
                mean: 0.85,
                median: 0.85,
              },
            ],
          },
        },
        species: {
          hist: {
            values: [
              {
                label: '1',
                count: 1,
              },
              {
                label: '0',
                count: 1,
              },
            ],
          },
        },
        large_petal_length: {
          hist: {
            values: [
              {
                label: '0',
                count: 2,
              },
            ],
          },
        },
        large_petal_width: {
          hist: {
            values: [
              {
                label: '0',
                count: 2,
              },
            ],
          },
        },
        step: {},
      },
      val: {
        sepal_length: {
          hist: {
            values: [
              {
                bin_start: 6.8,
                bin_end: 6.859999999999999,
                count: 1,
              },
              {
                bin_start: 6.859999999999999,
                bin_end: 6.92,
                count: 0,
              },
              {
                bin_start: 6.92,
                bin_end: 6.9799999999999995,
                count: 0,
              },
              {
                bin_start: 6.9799999999999995,
                bin_end: 7.04,
                count: 0,
              },
              {
                bin_start: 7.04,
                bin_end: 7.1,
                count: 0,
              },
              {
                bin_start: 7.1,
                bin_end: 7.16,
                count: 0,
              },
              {
                bin_start: 7.16,
                bin_end: 7.22,
                count: 0,
              },
              {
                bin_start: 7.22,
                bin_end: 7.28,
                count: 0,
              },
              {
                bin_start: 7.28,
                bin_end: 7.34,
                count: 0,
              },
              {
                bin_start: 7.34,
                bin_end: 7.4,
                count: 0,
              },
              {
                bin_start: 7.4,
                bin_end: 7.46,
                count: 0,
              },
              {
                bin_start: 7.46,
                bin_end: 7.5200000000000005,
                count: 0,
              },
              {
                bin_start: 7.5200000000000005,
                bin_end: 7.58,
                count: 0,
              },
              {
                bin_start: 7.58,
                bin_end: 7.640000000000001,
                count: 0,
              },
              {
                bin_start: 7.640000000000001,
                bin_end: 7.7,
                count: 1,
              },
              {
                min: 6.8,
                max: 7.7,
                mean: 7.25,
                median: 7.25,
              },
            ],
          },
        },
        sepal_width: {
          hist: {
            values: [
              {
                bin_start: 2.8,
                bin_end: 2.8666666666666667,
                count: 1,
              },
              {
                bin_start: 2.8666666666666667,
                bin_end: 2.933333333333333,
                count: 0,
              },
              {
                bin_start: 2.933333333333333,
                bin_end: 3,
                count: 0,
              },
              {
                bin_start: 3,
                bin_end: 3.0666666666666664,
                count: 0,
              },
              {
                bin_start: 3.0666666666666664,
                bin_end: 3.1333333333333333,
                count: 0,
              },
              {
                bin_start: 3.1333333333333333,
                bin_end: 3.1999999999999997,
                count: 0,
              },
              {
                bin_start: 3.1999999999999997,
                bin_end: 3.2666666666666666,
                count: 0,
              },
              {
                bin_start: 3.2666666666666666,
                bin_end: 3.333333333333333,
                count: 0,
              },
              {
                bin_start: 3.333333333333333,
                bin_end: 3.4,
                count: 0,
              },
              {
                bin_start: 3.4,
                bin_end: 3.4666666666666663,
                count: 0,
              },
              {
                bin_start: 3.4666666666666663,
                bin_end: 3.533333333333333,
                count: 0,
              },
              {
                bin_start: 3.533333333333333,
                bin_end: 3.5999999999999996,
                count: 0,
              },
              {
                bin_start: 3.5999999999999996,
                bin_end: 3.6666666666666665,
                count: 0,
              },
              {
                bin_start: 3.6666666666666665,
                bin_end: 3.7333333333333334,
                count: 0,
              },
              {
                bin_start: 3.7333333333333334,
                bin_end: 3.8,
                count: 1,
              },
              {
                min: 2.8,
                max: 3.8,
                mean: 3.3,
                median: 3.3,
              },
            ],
          },
        },
        petal_length: {
          hist: {
            values: [
              {
                bin_start: 4.8,
                bin_end: 4.926666666666667,
                count: 1,
              },
              {
                bin_start: 4.926666666666667,
                bin_end: 5.053333333333333,
                count: 0,
              },
              {
                bin_start: 5.053333333333333,
                bin_end: 5.18,
                count: 0,
              },
              {
                bin_start: 5.18,
                bin_end: 5.306666666666667,
                count: 0,
              },
              {
                bin_start: 5.306666666666667,
                bin_end: 5.433333333333334,
                count: 0,
              },
              {
                bin_start: 5.433333333333334,
                bin_end: 5.56,
                count: 0,
              },
              {
                bin_start: 5.56,
                bin_end: 5.6866666666666665,
                count: 0,
              },
              {
                bin_start: 5.6866666666666665,
                bin_end: 5.8133333333333335,
                count: 0,
              },
              {
                bin_start: 5.8133333333333335,
                bin_end: 5.9399999999999995,
                count: 0,
              },
              {
                bin_start: 5.9399999999999995,
                bin_end: 6.066666666666666,
                count: 0,
              },
              {
                bin_start: 6.066666666666666,
                bin_end: 6.193333333333333,
                count: 0,
              },
              {
                bin_start: 6.193333333333333,
                bin_end: 6.32,
                count: 0,
              },
              {
                bin_start: 6.32,
                bin_end: 6.446666666666666,
                count: 0,
              },
              {
                bin_start: 6.446666666666666,
                bin_end: 6.573333333333333,
                count: 0,
              },
              {
                bin_start: 6.573333333333333,
                bin_end: 6.7,
                count: 1,
              },
              {
                min: 4.8,
                max: 6.7,
                mean: 5.75,
                median: 5.75,
              },
            ],
          },
        },
        petal_width: {
          hist: {
            values: [
              {
                bin_start: 1.4,
                bin_end: 1.4533333333333334,
                count: 1,
              },
              {
                bin_start: 1.4533333333333334,
                bin_end: 1.5066666666666666,
                count: 0,
              },
              {
                bin_start: 1.5066666666666666,
                bin_end: 1.56,
                count: 0,
              },
              {
                bin_start: 1.56,
                bin_end: 1.6133333333333333,
                count: 0,
              },
              {
                bin_start: 1.6133333333333333,
                bin_end: 1.6666666666666667,
                count: 0,
              },
              {
                bin_start: 1.6666666666666667,
                bin_end: 1.72,
                count: 0,
              },
              {
                bin_start: 1.72,
                bin_end: 1.7733333333333334,
                count: 0,
              },
              {
                bin_start: 1.7733333333333334,
                bin_end: 1.8266666666666667,
                count: 0,
              },
              {
                bin_start: 1.8266666666666667,
                bin_end: 1.8800000000000001,
                count: 0,
              },
              {
                bin_start: 1.8800000000000001,
                bin_end: 1.9333333333333336,
                count: 0,
              },
              {
                bin_start: 1.9333333333333336,
                bin_end: 1.9866666666666668,
                count: 0,
              },
              {
                bin_start: 1.9866666666666668,
                bin_end: 2.04,
                count: 0,
              },
              {
                bin_start: 2.04,
                bin_end: 2.0933333333333337,
                count: 0,
              },
              {
                bin_start: 2.0933333333333337,
                bin_end: 2.146666666666667,
                count: 0,
              },
              {
                bin_start: 2.146666666666667,
                bin_end: 2.2,
                count: 1,
              },
              {
                min: 1.4,
                max: 2.2,
                mean: 1.8,
                median: 1.8,
              },
            ],
          },
        },
        species: {
          hist: {
            values: [
              {
                label: '2',
                count: 1,
              },
              {
                label: '1',
                count: 1,
              },
            ],
          },
        },
        large_petal_length: {
          hist: {
            values: [
              {
                label: '1',
                count: 1,
              },
              {
                label: '0',
                count: 1,
              },
            ],
          },
        },
        large_petal_width: {
          hist: {
            values: [
              {
                label: '1',
                count: 1,
              },
              {
                label: '0',
                count: 1,
              },
            ],
          },
        },
        step: {},
      },
    },
    dataUrl: 'datasets/73fbd8c90c3d3046c83aceab7d2ce8a2.csv',
    splitTarget: '60-20-20',
    splitActual: null,
    splitType: 'random',
    createdAt: '2023-09-25T16:58:41.791011+00:00',
    updatedAt: '2023-09-25T16:58:41.791014+00:00',
    createdById: 1,
    columnsMetadata: [
      {
        dataType: {
          domainKind: 'numeric',
          unit: 'mole',
        },
        description: 'egqfbemnyhrq',
        pattern: 'sepal_length',
        datasetId: 1,
      },
      {
        dataType: {
          domainKind: 'numeric',
          unit: 'mole',
        },
        description: 'gsqkmfwiwgoe',
        pattern: 'sepal_width',
        datasetId: 1,
      },
      {
        dataType: {
          domainKind: 'numeric',
          unit: 'mole',
        },
        description: 'ngxgfmamiwsr',
        pattern: 'petal_length',
        datasetId: 1,
      },
      {
        dataType: {
          domainKind: 'numeric',
          unit: 'mole',
        },
        description: 'juyravjmjwfi',
        pattern: 'petal_width',
        datasetId: 1,
      },
      {
        dataType: {
          domainKind: 'categorical',
          classes: {
            '0': 0,
            '1': 1,
            '2': 2,
          },
        },
        description: 'lmvojcloxykm',
        pattern: 'species',
        datasetId: 1,
      },
      {
        dataType: {
          domainKind: 'categorical',
          classes: {
            '0': 0,
            '1': 1,
          },
        },
        description: 'xgecsyhpcexh',
        pattern: 'large_petal_length',
        datasetId: 1,
      },
      {
        dataType: {
          domainKind: 'categorical',
          classes: {
            '0': 0,
            '1': 1,
          },
        },
        description: 'enlggptmadkh',
        pattern: 'large_petal_width',
        datasetId: 1,
      },
    ],
    readyStatus: 'ready',
    errors: null,
  },
  {
    id: 2,
    name: 'DATASET-CYPRESS',
    description: 'jmkpwmuxrmdlujoxlmgqgjod',
    rows: null,
    columns: null,
    bytes: 839,
    stats: null,
    dataUrl: 'datasets/a98c84d7ead3717e606a601c411c2917.csv',
    splitTarget: '80-10-10',
    splitActual: null,
    splitType: 'random',
    createdAt: '2023-09-29T16:51:40.063414+00:00',
    updatedAt: '2023-09-29T16:51:40.063417+00:00',
    createdById: 1,
    columnsMetadata: [
      {
        dataType: {
          domainKind: 'smiles',
        },
        description: 'a smile column',
        pattern: 'smiles',
        datasetId: 2,
      },
      {
        dataType: {
          domainKind: 'categorical',
          classes: {},
        },
        description: 'a categorical column',
        pattern: 'mwt_group',
        datasetId: 2,
      },
      {
        dataType: {
          domainKind: 'numeric',
          unit: 'mole',
        },
        description: 'another numerical column',
        pattern: 'tpsa',
        datasetId: 2,
      },
      {
        dataType: {
          domainKind: 'string',
        },
        description: '--',
        pattern: 'zinc_id',
        datasetId: 2,
      },
      {
        dataType: {
          domainKind: 'numeric',
          unit: 'mole',
        },
        description: 'a numerical column',
        pattern: 'mwt',
        datasetId: 2,
      },
    ],
    readyStatus: 'processing',
    errors: null,
  },
];
