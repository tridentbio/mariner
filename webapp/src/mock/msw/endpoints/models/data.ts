import { Model } from '@app/types/domain/models';

export const models: Partial<Model>[] = [
  {
    id: 1,
    name: 'SOME_MODEL_NAME',
    mlflowName: '1-SOME_MODEL_NAME-c99289aa-f6d3-4c61-99f1-2a8e3cd24dd7',
    description: 'cqlqrats',
    createdById: 1,
    createdBy: {
      email: 'admin@mariner.trident.bio',
      isActive: true,
      isSuperuser: false,
      fullName: undefined,
      id: 1,
    },
    datasetId: 1,
    dataset: {
      id: 1,
      name: 'IRIS_DATASET_NAME',
      description: 'rhuxbedsfawiyrddnoicssmc',
      rows: 10,
      columns: 8,
      bytes: 340,
      dataUrl: 'datasets/73fbd8c90c3d3046c83aceab7d2ce8a2.csv',
      splitTarget: '60-20-20',
      splitActual: undefined,
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
      errors: undefined,
    },
    versions: [
      {
        id: 1,
        modelId: 1,
        name: 'test model version',
        description: 'fwscttrs',
        mlflowVersion: '25',
        mlflowModelName:
          '1-SOME_MODEL_NAME-c99289aa-f6d3-4c61-99f1-2a8e3cd24dd7',
        config: {
          name: 'test model version',
          framework: 'torch',
          spec: {
            layers: [
              {
                type: 'fleet.model_builder.layers.Concat',
                name: 'Concat-0',
                constructorArgs: {
                  dim: 1,
                },
                forwardArgs: {
                  xs: ['$sepal_length', '$sepal_width'],
                },
              },
              {
                type: 'torch.nn.Linear',
                name: 'Linear-1',
                constructorArgs: {
                  in_features: 2,
                  out_features: 16,
                  bias: true,
                },
                forwardArgs: {
                  input: '$Concat-0',
                },
              },
              {
                type: 'torch.nn.ReLU',
                name: 'ReLU-2',
                constructorArgs: {
                  inplace: false,
                },
                forwardArgs: {
                  input: '$Linear-1',
                },
              },
              {
                type: 'torch.nn.Linear',
                name: 'Linear-3',
                constructorArgs: {
                  in_features: 16,
                  out_features: 16,
                  bias: true,
                },
                forwardArgs: {
                  input: '$ReLU-2',
                },
              },
              {
                type: 'torch.nn.ReLU',
                name: 'ReLU-4',
                constructorArgs: {
                  inplace: false,
                },
                forwardArgs: {
                  input: '$Linear-3',
                },
              },
              {
                type: 'torch.nn.Linear',
                name: 'Linear-5',
                constructorArgs: {
                  in_features: 16,
                  out_features: 1,
                  bias: true,
                },
                forwardArgs: {
                  input: '$ReLU-4',
                },
              },
            ],
          },
          dataset: {
            name: 'IRIS_DATASET_NAME',
            strategy: 'forwardArgs',
            targetColumns: [
              {
                name: 'large_petal_length',
                dataType: {
                  domainKind: 'categorical',
                  classes: {
                    '0': 0,
                    '1': 1,
                  },
                },
                outModule: 'Linear-5',
                lossFn: 'torch.nn.BCEWithLogitsLoss',
                columnType: 'binary',
              },
            ],
            featureColumns: [
              {
                name: 'sepal_length',
                dataType: {
                  domainKind: 'numeric',
                  unit: 'mole',
                },
              },
              {
                name: 'sepal_width',
                dataType: {
                  domainKind: 'numeric',
                  unit: 'mole',
                },
              },
            ],
            featurizers: [],
            transforms: [],
          },
        },
        createdAt: '2023-09-25T16:58:49.069842+00:00',
        updatedAt: '2023-09-25T16:58:49.069842',
      },
    ],
    columns: [
      {
        modelId: 1,
        columnName: 'sepal_length',
        columnType: 'feature',
      },
      {
        modelId: 1,
        columnName: 'sepal_width',
        columnType: 'feature',
      },
      {
        modelId: 1,
        columnName: 'large_petal_length',
        columnType: 'target',
      },
    ],
    createdAt: '2023-09-25T16:58:48.179663',
    updatedAt: '2023-09-25T16:58:48.179663',
  },
  {
    id: 2,
    name: 'xcpakbes',
    mlflowName: '1-xcpakbes-8d16908f-de8a-498b-a538-ac3034ab243f',
    description: 'cmyfnnsw',
    createdById: 1,
    createdBy: {
      email: 'admin@mariner.trident.bio',
      isActive: true,
      isSuperuser: false,
      fullName: undefined,
      id: 1,
    },
    datasetId: 1,
    dataset: {
      id: 1,
      name: 'IRIS_DATASET_NAME',
      description: 'rhuxbedsfawiyrddnoicssmc',
      rows: 10,
      columns: 8,
      bytes: 340,
      dataUrl: 'datasets/73fbd8c90c3d3046c83aceab7d2ce8a2.csv',
      splitTarget: '60-20-20',
      splitActual: undefined,
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
      errors: undefined,
    },
    versions: [
      {
        id: 2,
        modelId: 2,
        name: 'ysjoyejs',
        description: 'aqabeota',
        mlflowVersion: undefined,
        mlflowModelName: '1-xcpakbes-8d16908f-de8a-498b-a538-ac3034ab243f',
        config: {
          framework: 'sklearn',
          name: 'ysjoyejs',
          dataset: {
            name: 'IRIS_DATASET_NAME',
            strategy: 'pipeline',
            targetColumns: [
              {
                name: 'species',
                dataType: {
                  domainKind: 'categorical',
                  classes: {
                    '0': 0,
                    '1': 1,
                    '2': 2,
                  },
                },
                transforms: [],
                featurizers: [
                  {
                    type: 'sklearn.preprocessing.OneHotEncoder',
                    constructorArgs: undefined,
                  },
                ],
              },
            ],
            featureColumns: [
              {
                name: 'sepal_length',
                dataType: {
                  domainKind: 'numeric',
                },
                transforms: [
                  {
                    type: 'sklearn.preprocessing.StandardScaler',
                    constructorArgs: {},
                  },
                ],
                featurizers: [],
              },
              {
                name: 'sepal_width',
                dataType: {
                  domainKind: 'numeric',
                },
                transforms: [],
                featurizers: [],
              },
            ],
          },
          spec: {
            model: {
              type: 'sklearn.ensemble.RandomForestClassifier',
              taskType: ['multiclass'],
              constructorArgs: {
                n_estimators: 100,
                criterion: 'entropy',
              },
              fitArgs: undefined,
            },
          },
        },
        createdAt: '2023-10-02T16:17:45.731449+00:00',
        updatedAt: '2023-10-02T16:17:45.731449',
      },
    ],
    columns: [
      {
        modelId: 2,
        columnName: 'sepal_length',
        columnType: 'feature',
      },
      {
        modelId: 2,
        columnName: 'sepal_width',
        columnType: 'feature',
      },
      {
        modelId: 2,
        columnName: 'species',
        columnType: 'target',
      },
    ],
    createdAt: '2023-10-02T16:17:42.193525',
    updatedAt: '2023-10-02T16:17:42.193525',
  },
];

export const modelOptionsData = [
  {
    classPath: 'fleet.model_builder.layers.OneHot',
    component: {
      type: 'fleet.model_builder.layers.OneHot',
      constructorArgsSummary: {},
      forwardArgsSummary: {
        x1: 'typing.Union[list[str], list[int]]',
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>A helper layer that outputs the one-hot encoding representation of\nit’s categorical inputs</p>\n</div></blockquote>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {},
  },
  {
    classPath: 'fleet.model_builder.layers.GlobalPooling',
    component: {
      type: 'fleet.model_builder.layers.GlobalPooling',
      constructorArgsSummary: {
        aggr: "<class 'str'>?",
      },
      forwardArgsSummary: {
        x: "<class 'torch.Tensor'>",
        batch: 'typing.Optional[torch.Tensor]?',
        size: 'typing.Optional[int]?',
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <dl>\n<dt>A global pooling module that wraps the usage of</dt><dd><p><code class="xref py py-meth docutils literal notranslate"><span class="pre">global_add_pool()</span></code>,\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">global_mean_pool()</span></code> and\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">global_max_pool()</span></code> into a single module.</p>\n<dl class="simple">\n<dt>Args:</dt><dd><dl class="simple">\n<dt>aggr (string or List[str]): The aggregation scheme to use</dt><dd><p>(<code class="xref py py-obj docutils literal notranslate"><span class="pre">&quot;add&quot;</span></code>, <code class="xref py py-obj docutils literal notranslate"><span class="pre">&quot;mean&quot;</span></code>, <code class="xref py py-obj docutils literal notranslate"><span class="pre">&quot;max&quot;</span></code>).\nIf given as a list, will make use of multiple aggregations in which\ndifferent outputs will get concatenated in the last dimension.</p>\n</dd>\n</dl>\n</dd>\n</dl>\n</dd>\n</dl>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {
      aggr: 'add',
    },
  },
  {
    classPath: 'fleet.model_builder.layers.Concat',
    component: {
      type: 'fleet.model_builder.layers.Concat',
      constructorArgsSummary: {
        dim: "<class 'int'>?",
      },
      forwardArgsSummary: {
        xs: 'typing.List[torch.Tensor]',
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>(Based on torch.cat)</p>\n<p>cat(tensors, dim=0, <a href="#id1"><span class="problematic" id="id2">*</span></a>, out=None) -&gt; Tensor</p>\n<p>Concatenates the given sequence of <code class="xref py py-attr docutils literal notranslate"><span class="pre">seq</span></code> tensors in the given dimension.\nAll tensors must either have the same shape (except in the concatenating\ndimension) or be empty.</p>\n<p><code class="xref py py-func docutils literal notranslate"><span class="pre">torch.cat()</span></code> can be seen as an inverse operation for <code class="xref py py-func docutils literal notranslate"><span class="pre">torch.split()</span></code>\nand <code class="xref py py-func docutils literal notranslate"><span class="pre">torch.chunk()</span></code>.</p>\n<p><code class="xref py py-func docutils literal notranslate"><span class="pre">torch.cat()</span></code> can be best understood via examples.</p>\n<dl>\n<dt>Args:</dt><dd><dl class="simple">\n<dt>tensors (sequence of Tensors): any python sequence of tensors of the same type.</dt><dd><p>Non-empty tensors provided must have the same shape, except in the\ncat dimension.</p>\n</dd>\n</dl>\n<p>dim (int, optional): the dimension over which the tensors are concatenated</p>\n</dd>\n<dt>Keyword args:</dt><dd><p>out (Tensor, optional): the output tensor.</p>\n</dd>\n</dl>\n<p>Example:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">x</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">x</span>\n<span class="go">tensor([[ 0.6580, -1.0969, -0.4614],</span>\n<span class="go">        [-0.1034, -0.5790,  0.1497]])</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">((</span><span class="n">x</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">x</span><span class="p">),</span> <span class="mi">0</span><span class="p">)</span>\n<span class="go">tensor([[ 0.6580, -1.0969, -0.4614],</span>\n<span class="go">        [-0.1034, -0.5790,  0.1497],</span>\n<span class="go">        [ 0.6580, -1.0969, -0.4614],</span>\n<span class="go">        [-0.1034, -0.5790,  0.1497],</span>\n<span class="go">        [ 0.6580, -1.0969, -0.4614],</span>\n<span class="go">        [-0.1034, -0.5790,  0.1497]])</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">((</span><span class="n">x</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">x</span><span class="p">),</span> <span class="mi">1</span><span class="p">)</span>\n<span class="go">tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,</span>\n<span class="go">         -1.0969, -0.4614],</span>\n<span class="go">        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,</span>\n<span class="go">         -0.5790,  0.1497]])</span>\n</pre></div>\n</div>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {
      dim: 0,
    },
  },
  {
    classPath: 'fleet.model_builder.layers.AddPooling',
    component: {
      type: 'fleet.model_builder.layers.AddPooling',
      constructorArgsSummary: {
        dim: 'typing.Optional[int]?',
      },
      forwardArgsSummary: {
        x: "<class 'torch.Tensor'>",
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>Returns the summation of it’s inputs (using tensor.sum(dim=dim))</p>\n<dl class="simple">\n<dt>See:</dt><dd><p>&lt;<a class="reference external" href="https://pytorch.org/docs/stable/generated/torch.sum.html">https://pytorch.org/docs/stable/generated/torch.sum.html</a>&gt;</p>\n</dd>\n</dl>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      dim: null,
    },
  },
  {
    classPath: 'torch.nn.Linear',
    component: {
      type: 'torch.nn.Linear',
      constructorArgsSummary: {
        in_features: "<class 'int'>",
        out_features: "<class 'int'>",
        bias: "<class 'bool'>?",
      },
      forwardArgsSummary: {
        input: "<class 'torch.Tensor'>",
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink:
      'https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear',
    docs: '<div class="docstring">\n    \n  <p>Applies a linear transformation to the incoming data: <span class="math notranslate nohighlight">y = xA^T + b</span></p>\n<blockquote>\n<div><p>This module supports <span class="xref std std-ref">TensorFloat32</span>.</p>\n<p>On certain ROCm devices, when using float16 inputs this module will use <span class="xref std std-ref">different precision</span> for backward.</p>\n<dl>\n<dt>Args:</dt><dd><p>in_features: size of each input sample\nout_features: size of each output sample\nbias: If set to <code class="docutils literal notranslate"><span class="pre">False</span></code>, the layer will not learn an additive bias.</p>\n<blockquote>\n<div><p>Default: <code class="docutils literal notranslate"><span class="pre">True</span></code></p>\n</div></blockquote>\n</dd>\n<dt>Shape:</dt><dd><ul class="simple">\n<li><p>Input: <span class="math notranslate nohighlight">(*, H_{in})</span> where <span class="math notranslate nohighlight">*</span> means any number of\ndimensions including none and <span class="math notranslate nohighlight">H_{in} = \\text{in\\_features}</span>.</p></li>\n<li><p>Output: <span class="math notranslate nohighlight">(*, H_{out})</span> where all but the last dimension\nare the same shape as the input and <span class="math notranslate nohighlight">H_{out} = \\text{out\\_features}</span>.</p></li>\n</ul>\n</dd>\n<dt>Attributes:</dt><dd><dl class="simple">\n<dt>weight: the learnable weights of the module of shape</dt><dd><p><span class="math notranslate nohighlight">(\\text{out\\_features}, \\text{in\\_features})</span>. The values are\ninitialized from <span class="math notranslate nohighlight">\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})</span>, where\n<span class="math notranslate nohighlight">k = \\frac{1}{\\text{in\\_features}}</span></p>\n</dd>\n<dt>bias:   the learnable bias of the module of shape <span class="math notranslate nohighlight">(\\text{out\\_features})</span>.</dt><dd><p>If <code class="xref py py-attr docutils literal notranslate"><span class="pre">bias</span></code> is <code class="docutils literal notranslate"><span class="pre">True</span></code>, the values are initialized from\n<span class="math notranslate nohighlight">\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})</span> where\n<span class="math notranslate nohighlight">k = \\frac{1}{\\text{in\\_features}}</span></p>\n</dd>\n</dl>\n</dd>\n</dl>\n<p>Examples:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">m</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">20</span><span class="p">,</span> <span class="mi">30</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">input</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">output</span> <span class="o">=</span> <span class="n">m</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">output</span><span class="o">.</span><span class="n">size</span><span class="p">())</span>\n<span class="go">torch.Size([128, 30])</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {
      bias: true,
    },
  },
  {
    classPath: 'torch.nn.Sigmoid',
    component: {
      type: 'torch.nn.Sigmoid',
      constructorArgsSummary: {},
      forwardArgsSummary: {
        input: "<class 'torch.Tensor'>",
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink:
      'https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Sigmoid',
    docs: '<div class="docstring">\n    \n  <p>Applies the element-wise function:</p>\n<blockquote>\n<div><div class="math notranslate nohighlight">\n\\text{Sigmoid}(x) = \\sigma(x) = \\frac{1}{1 + \\exp(-x)}</div>\n<dl class="simple">\n<dt>Shape:</dt><dd><ul class="simple">\n<li><p>Input: <span class="math notranslate nohighlight">(*)</span>, where <span class="math notranslate nohighlight">*</span> means any number of dimensions.</p></li>\n<li><p>Output: <span class="math notranslate nohighlight">(*)</span>, same shape as the input.</p></li>\n</ul>\n</dd>\n</dl>\n\n<p>Examples:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">m</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Sigmoid</span><span class="p">()</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">input</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">output</span> <span class="o">=</span> <span class="n">m</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {},
  },
  {
    classPath: 'torch.nn.ReLU',
    component: {
      type: 'torch.nn.ReLU',
      constructorArgsSummary: {
        inplace: "<class 'bool'>?",
      },
      forwardArgsSummary: {
        input: "<class 'torch.Tensor'>",
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink:
      'https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.ReLU',
    docs: '<div class="docstring">\n    \n  <p>Applies the rectified linear unit function element-wise:</p>\n<blockquote>\n<div><p><span class="math notranslate nohighlight">\\text{ReLU}(x) = (x)^+ = \\max(0, x)</span></p>\n<dl class="simple">\n<dt>Args:</dt><dd><p>inplace: can optionally do the operation in-place. Default: <code class="docutils literal notranslate"><span class="pre">False</span></code></p>\n</dd>\n<dt>Shape:</dt><dd><ul class="simple">\n<li><p>Input: <span class="math notranslate nohighlight">(*)</span>, where <span class="math notranslate nohighlight">*</span> means any number of dimensions.</p></li>\n<li><p>Output: <span class="math notranslate nohighlight">(*)</span>, same shape as the input.</p></li>\n</ul>\n</dd>\n</dl>\n\n<p>Examples:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span>  <span class="o">&gt;&gt;&gt;</span> <span class="n">m</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">()</span>\n  <span class="o">&gt;&gt;&gt;</span> <span class="nb">input</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>\n  <span class="o">&gt;&gt;&gt;</span> <span class="n">output</span> <span class="o">=</span> <span class="n">m</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>\n\n\n<span class="n">An</span> <span class="n">implementation</span> <span class="n">of</span> <span class="n">CReLU</span> <span class="o">-</span> <span class="n">https</span><span class="p">:</span><span class="o">//</span><span class="n">arxiv</span><span class="o">.</span><span class="n">org</span><span class="o">/</span><span class="nb">abs</span><span class="o">/</span><span class="mf">1603.05201</span>\n\n  <span class="o">&gt;&gt;&gt;</span> <span class="n">m</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">()</span>\n  <span class="o">&gt;&gt;&gt;</span> <span class="nb">input</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>\n  <span class="o">&gt;&gt;&gt;</span> <span class="n">output</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">((</span><span class="n">m</span><span class="p">(</span><span class="nb">input</span><span class="p">),</span> <span class="n">m</span><span class="p">(</span><span class="o">-</span><span class="nb">input</span><span class="p">)))</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {
      inplace: false,
    },
  },
  {
    classPath: 'torch_geometric.nn.GCNConv',
    component: {
      type: 'torch_geometric.nn.GCNConv',
      constructorArgsSummary: {
        in_channels: "<class 'int'>",
        out_channels: "<class 'int'>",
        improved: "<class 'bool'>?",
        cached: "<class 'bool'>?",
        add_self_loops: "<class 'bool'>?",
        normalize: "<class 'bool'>?",
        bias: "<class 'bool'>?",
      },
      forwardArgsSummary: {
        x: "<class 'torch.Tensor'>",
        edge_index:
          'typing.Union[torch.Tensor, torch_sparse.tensor.SparseTensor]',
        edge_weight: 'typing.Optional[torch.Tensor]?',
      },
    },
    type: 'layer',
    argsOptions: null,
    docsLink:
      'https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.GCNConv',
    docs: '<div class="docstring">\n    \n  <dl>\n<dt>The graph convolutional operator from the <a href="#id1"><span class="problematic" id="id2">`</span></a>”Semi-supervised</dt><dd><p>Classification with Graph Convolutional Networks”\n&lt;<a class="reference external" href="https://arxiv.org/abs/1609.02907">https://arxiv.org/abs/1609.02907</a>&gt;`_ paper</p>\n<div class="math notranslate nohighlight">\n\\mathbf{X}^{\\prime} = \\mathbf{\\hat{D}}^{-1/2} \\mathbf{\\hat{A}}\n\\mathbf{\\hat{D}}^{-1/2} \\mathbf{X} \\mathbf{\\Theta},</div>\n<p>where <span class="math notranslate nohighlight">\\mathbf{\\hat{A}} = \\mathbf{A} + \\mathbf{I}</span> denotes the\nadjacency matrix with inserted self-loops and\n<span class="math notranslate nohighlight">\\hat{D}_{ii} = \\sum_{j=0} \\hat{A}_{ij}</span> its diagonal degree matrix.\nThe adjacency matrix can include other values than <code class="xref py py-obj docutils literal notranslate"><span class="pre">1</span></code> representing\nedge weights via the optional <code class="xref py py-obj docutils literal notranslate"><span class="pre">edge_weight</span></code> tensor.</p>\n<p>Its node-wise formulation is given by:</p>\n<div class="math notranslate nohighlight">\n\\mathbf{x}^{\\prime}_i = \\mathbf{\\Theta}^{\\top} \\sum_{j \\in\n\\mathcal{N}(v) \\cup \\{ i \\}} \\frac{e_{j,i}}{\\sqrt{\\hat{d}_j\n\\hat{d}_i}} \\mathbf{x}_j</div>\n<p>with <span class="math notranslate nohighlight">\\hat{d}_i = 1 + \\sum_{j \\in \\mathcal{N}(i)} e_{j,i}</span>, where\n<span class="math notranslate nohighlight">e_{j,i}</span> denotes the edge weight from source node <code class="xref py py-obj docutils literal notranslate"><span class="pre">j</span></code> to target\nnode <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code> (default: <code class="xref py py-obj docutils literal notranslate"><span class="pre">1.0</span></code>)</p>\n<dl>\n<dt>Args:</dt><dd><dl class="simple">\n<dt>in_channels (int): Size of each input sample, or <code class="xref py py-obj docutils literal notranslate"><span class="pre">-1</span></code> to derive</dt><dd><p>the size from the first input(s) to the forward method.</p>\n</dd>\n</dl>\n<p>out_channels (int): Size of each output sample.\nimproved (bool, optional): If set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">True</span></code>, the layer computes</p>\n<blockquote>\n<div><p><span class="math notranslate nohighlight">\\mathbf{\\hat{A}}</span> as <span class="math notranslate nohighlight">\\mathbf{A} + 2\\mathbf{I}</span>.\n(default: <code class="xref py py-obj docutils literal notranslate"><span class="pre">False</span></code>)</p>\n</div></blockquote>\n<dl class="simple">\n<dt>cached (bool, optional): If set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">True</span></code>, the layer will cache</dt><dd><p>the computation of <span class="math notranslate nohighlight">\\mathbf{\\hat{D}}^{-1/2} \\mathbf{\\hat{A}}\n\\mathbf{\\hat{D}}^{-1/2}</span> on first execution, and will use the\ncached version for further executions.\nThis parameter should only be set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">True</span></code> in transductive\nlearning scenarios. (default: <code class="xref py py-obj docutils literal notranslate"><span class="pre">False</span></code>)</p>\n</dd>\n<dt>add_self_loops (bool, optional): If set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">False</span></code>, will not add</dt><dd><p>self-loops to the input graph. (default: <code class="xref py py-obj docutils literal notranslate"><span class="pre">True</span></code>)</p>\n</dd>\n<dt>normalize (bool, optional): Whether to add self-loops and compute</dt><dd><p>symmetric normalization coefficients on the fly.\n(default: <code class="xref py py-obj docutils literal notranslate"><span class="pre">True</span></code>)</p>\n</dd>\n<dt>bias (bool, optional): If set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">False</span></code>, the layer will not learn</dt><dd><p>an additive bias. (default: <code class="xref py py-obj docutils literal notranslate"><span class="pre">True</span></code>)</p>\n</dd>\n<dt><a href="#id3"><span class="problematic" id="id4">**</span></a>kwargs (optional): Additional arguments of</dt><dd><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch_geometric.nn.conv.MessagePassing</span></code>.</p>\n</dd>\n</dl>\n</dd>\n<dt>Shapes:</dt><dd><ul class="simple">\n<li><p><strong>input:</strong>\nnode features <span class="math notranslate nohighlight">(|\\mathcal{V}|, F_{in})</span>,\nedge indices <span class="math notranslate nohighlight">(2, |\\mathcal{E}|)</span>,\nedge weights <span class="math notranslate nohighlight">(|\\mathcal{E}|)</span> <em>(optional)</em></p></li>\n<li><p><strong>output:</strong> node features <span class="math notranslate nohighlight">(|\\mathcal{V}|, F_{out})</span></p></li>\n</ul>\n</dd>\n</dl>\n</dd>\n</dl>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {
      improved: false,
      cached: false,
      add_self_loops: true,
      normalize: true,
      bias: true,
    },
  },
  {
    classPath: 'torch.nn.Embedding',
    component: {
      type: 'torch.nn.Embedding',
      constructorArgsSummary: {
        num_embeddings: "<class 'int'>",
        embedding_dim: "<class 'int'>",
        padding_idx: 'typing.Optional[int]?',
        max_norm: 'typing.Optional[float]?',
        norm_type: "<class 'float'>?",
        scale_grad_by_freq: "<class 'bool'>?",
        sparse: "<class 'bool'>?",
      },
      forwardArgsSummary: {
        input: "<class 'torch.Tensor'>",
      },
    },
    type: 'layer',
    argsOptions: {
      max_norm: [],
      norm_type: [],
    },
    docsLink:
      'https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Embedding',
    docs: '<div class="docstring">\n    \n  <p>A simple lookup table that stores embeddings of a fixed dictionary and size.</p>\n<blockquote>\n<div><p>This module is often used to store word embeddings and retrieve them using indices.\nThe input to the module is a list of indices, and the output is the corresponding\nword embeddings.</p>\n<dl>\n<dt>Args:</dt><dd><p>num_embeddings (int): size of the dictionary of embeddings\nembedding_dim (int): the size of each embedding vector\npadding_idx (int, optional): If specified, the entries at <code class="xref py py-attr docutils literal notranslate"><span class="pre">padding_idx</span></code> do not contribute to the gradient;</p>\n<blockquote>\n<div><p>therefore, the embedding vector at <code class="xref py py-attr docutils literal notranslate"><span class="pre">padding_idx</span></code> is not updated during training,\ni.e. it remains as a fixed “pad”. For a newly constructed Embedding,\nthe embedding vector at <code class="xref py py-attr docutils literal notranslate"><span class="pre">padding_idx</span></code> will default to all zeros,\nbut can be updated to another value to be used as the padding vector.</p>\n</div></blockquote>\n<dl class="simple">\n<dt>max_norm (float, optional): If given, each embedding vector with norm larger than <code class="xref py py-attr docutils literal notranslate"><span class="pre">max_norm</span></code></dt><dd><p>is renormalized to have norm <code class="xref py py-attr docutils literal notranslate"><span class="pre">max_norm</span></code>.</p>\n</dd>\n</dl>\n<p>norm_type (float, optional): The p of the p-norm to compute for the <code class="xref py py-attr docutils literal notranslate"><span class="pre">max_norm</span></code> option. Default <code class="docutils literal notranslate"><span class="pre">2</span></code>.\nscale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of</p>\n<blockquote>\n<div><p>the words in the mini-batch. Default <code class="docutils literal notranslate"><span class="pre">False</span></code>.</p>\n</div></blockquote>\n<dl class="simple">\n<dt>sparse (bool, optional): If <code class="docutils literal notranslate"><span class="pre">True</span></code>, gradient w.r.t. <code class="xref py py-attr docutils literal notranslate"><span class="pre">weight</span></code> matrix will be a sparse tensor.</dt><dd><p>See Notes for more details regarding sparse gradients.</p>\n</dd>\n</dl>\n</dd>\n<dt>Attributes:</dt><dd><dl class="simple">\n<dt>weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)</dt><dd><p>initialized from <span class="math notranslate nohighlight">\\mathcal{N}(0, 1)</span></p>\n</dd>\n</dl>\n</dd>\n<dt>Shape:</dt><dd><ul class="simple">\n<li><p>Input: <span class="math notranslate nohighlight">(*)</span>, IntTensor or LongTensor of arbitrary shape containing the indices to extract</p></li>\n<li><p>Output: <span class="math notranslate nohighlight">(*, H)</span>, where <cite>*</cite> is the input shape and <span class="math notranslate nohighlight">H=\\text{embedding\\_dim}</span></p></li>\n</ul>\n</dd>\n</dl>\n<div class="admonition note">\n<p class="admonition-title">Note</p>\n<p>Keep in mind that only a limited number of optimizers support\nsparse gradients: currently it’s <code class="xref py py-class docutils literal notranslate"><span class="pre">optim.SGD</span></code> (<cite>CUDA</cite> and <cite>CPU</cite>),\n<code class="xref py py-class docutils literal notranslate"><span class="pre">optim.SparseAdam</span></code> (<cite>CUDA</cite> and <cite>CPU</cite>) and <code class="xref py py-class docutils literal notranslate"><span class="pre">optim.Adagrad</span></code> (<cite>CPU</cite>)</p>\n</div>\n<div class="admonition note">\n<p class="admonition-title">Note</p>\n<p>When <code class="xref py py-attr docutils literal notranslate"><span class="pre">max_norm</span></code> is not <code class="docutils literal notranslate"><span class="pre">None</span></code>, <code class="xref py py-class docutils literal notranslate"><span class="pre">Embedding</span></code>’s forward method will modify the\n<code class="xref py py-attr docutils literal notranslate"><span class="pre">weight</span></code> tensor in-place. Since tensors needed for gradient computations cannot be\nmodified in-place, performing a differentiable operation on <code class="docutils literal notranslate"><span class="pre">Embedding.weight</span></code> before\ncalling <code class="xref py py-class docutils literal notranslate"><span class="pre">Embedding</span></code>’s forward method requires cloning <code class="docutils literal notranslate"><span class="pre">Embedding.weight</span></code> when\n<code class="xref py py-attr docutils literal notranslate"><span class="pre">max_norm</span></code> is not <code class="docutils literal notranslate"><span class="pre">None</span></code>. For example:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">n</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">m</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">7</span>\n<span class="n">embedding</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Embedding</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">max_norm</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>\n<span class="n">W</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">((</span><span class="n">m</span><span class="p">,</span> <span class="n">d</span><span class="p">),</span> <span class="n">requires_grad</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>\n<span class="n">idx</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>\n<span class="n">a</span> <span class="o">=</span> <span class="n">embedding</span><span class="o">.</span><span class="n">weight</span><span class="o">.</span><span class="n">clone</span><span class="p">()</span> <span class="o">@</span> <span class="n">W</span><span class="o">.</span><span class="n">t</span><span class="p">()</span>  <span class="c1"># weight must be cloned for this to be differentiable</span>\n<span class="n">b</span> <span class="o">=</span> <span class="n">embedding</span><span class="p">(</span><span class="n">idx</span><span class="p">)</span> <span class="o">@</span> <span class="n">W</span><span class="o">.</span><span class="n">t</span><span class="p">()</span>  <span class="c1"># modifies weight in-place</span>\n<span class="n">out</span> <span class="o">=</span> <span class="p">(</span><span class="n">a</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span> <span class="o">+</span> <span class="n">b</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="mi">1</span><span class="p">))</span>\n<span class="n">loss</span> <span class="o">=</span> <span class="n">out</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">()</span><span class="o">.</span><span class="n">prod</span><span class="p">()</span>\n<span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>\n</pre></div>\n</div>\n</div>\n<p>Examples:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="c1"># an Embedding module containing 10 tensors of size 3</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">embedding</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Embedding</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="c1"># a batch of 2 samples of 4 indices each</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">input</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">LongTensor</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">5</span><span class="p">],</span> <span class="p">[</span><span class="mi">4</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">9</span><span class="p">]])</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="c1"># xdoctest: +IGNORE_WANT(&quot;non-deterministic&quot;)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">embedding</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>\n<span class="go">tensor([[[-0.0251, -1.6902,  0.7172],</span>\n<span class="go">         [-0.6431,  0.0748,  0.6969],</span>\n<span class="go">         [ 1.4970,  1.3448, -0.9685],</span>\n<span class="go">         [-0.3677, -2.7265, -0.1685]],</span>\n\n<span class="go">        [[ 1.4970,  1.3448, -0.9685],</span>\n<span class="go">         [ 0.4362, -0.4004,  0.9400],</span>\n<span class="go">         [-0.6431,  0.0748,  0.6969],</span>\n<span class="go">         [ 0.9124, -2.3616,  1.1151]]])</span>\n\n\n<span class="gp">&gt;&gt;&gt; </span><span class="c1"># example with padding_idx</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">embedding</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Embedding</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">padding_idx</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">input</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">LongTensor</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">5</span><span class="p">]])</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">embedding</span><span class="p">(</span><span class="nb">input</span><span class="p">)</span>\n<span class="go">tensor([[[ 0.0000,  0.0000,  0.0000],</span>\n<span class="go">         [ 0.1535, -2.0309,  0.9315],</span>\n<span class="go">         [ 0.0000,  0.0000,  0.0000],</span>\n<span class="go">         [-0.1655,  0.9897,  0.0635]]])</span>\n\n<span class="gp">&gt;&gt;&gt; </span><span class="c1"># example of changing `pad` vector</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">padding_idx</span> <span class="o">=</span> <span class="mi">0</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">embedding</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Embedding</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">padding_idx</span><span class="o">=</span><span class="n">padding_idx</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">embedding</span><span class="o">.</span><span class="n">weight</span>\n<span class="go">Parameter containing:</span>\n<span class="go">tensor([[ 0.0000,  0.0000,  0.0000],</span>\n<span class="go">        [-0.7895, -0.7089, -0.0364],</span>\n<span class="go">        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>\n<span class="gp">... </span>    <span class="n">embedding</span><span class="o">.</span><span class="n">weight</span><span class="p">[</span><span class="n">padding_idx</span><span class="p">]</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">embedding</span><span class="o">.</span><span class="n">weight</span>\n<span class="go">Parameter containing:</span>\n<span class="go">tensor([[ 1.0000,  1.0000,  1.0000],</span>\n<span class="go">        [-0.7895, -0.7089, -0.0364],</span>\n<span class="go">        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {
      padding_idx: null,
      max_norm: 1,
      norm_type: 2,
      scale_grad_by_freq: false,
      sparse: false,
    },
  },
  {
    classPath: 'torch.nn.TransformerEncoderLayer',
    component: {
      type: 'torch.nn.TransformerEncoderLayer',
      constructorArgsSummary: {
        d_model: "<class 'int'>",
        nhead: "<class 'int'>",
        dim_feedforward: "<class 'int'>?",
        dropout: "<class 'float'>?",
        activation:
          'typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]?',
        layer_norm_eps: "<class 'float'>?",
        batch_first: "<class 'bool'>?",
        norm_first: "<class 'bool'>?",
      },
      forwardArgsSummary: {
        src: "<class 'torch.Tensor'>",
        src_mask: 'typing.Optional[torch.Tensor]?',
        src_key_padding_mask: 'typing.Optional[torch.Tensor]?',
        is_causal: "<class 'bool'>?",
      },
    },
    type: 'layer',
    argsOptions: {
      activation: [
        {
          key: 'relu',
          label: 'relu',
          latex: '...',
        },
        {
          key: 'sigmoid',
          label: 'sigmoid',
          latex: '...',
        },
      ],
    },
    docsLink:
      'https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.TransformerEncoderLayer',
    docs: '<div class="docstring">\n    \n  <dl>\n<dt>TransformerEncoderLayer is made up of self-attn and feedforward network.</dt><dd><p>This standard encoder layer is based on the paper “Attention Is All You Need”.\nAshish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,\nLukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in\nNeural Information Processing Systems, pages 6000-6010. Users may modify or implement\nin a different way during application.</p>\n<dl>\n<dt>Args:</dt><dd><p>d_model: the number of expected features in the input (required).\nnhead: the number of heads in the multiheadattention models (required).\ndim_feedforward: the dimension of the feedforward network model (default=2048).\ndropout: the dropout value (default=0.1).\nactivation: the activation function of the intermediate layer, can be a string</p>\n<blockquote>\n<div><p>(“relu” or “gelu”) or a unary callable. Default: relu</p>\n</div></blockquote>\n<p>layer_norm_eps: the eps value in layer normalization components (default=1e-5).\nbatch_first: If <code class="docutils literal notranslate"><span class="pre">True</span></code>, then the input and output tensors are provided</p>\n<blockquote>\n<div><p>as (batch, seq, feature). Default: <code class="docutils literal notranslate"><span class="pre">False</span></code> (seq, batch, feature).</p>\n</div></blockquote>\n<dl class="simple">\n<dt>norm_first: if <code class="docutils literal notranslate"><span class="pre">True</span></code>, layer norm is done prior to attention and feedforward</dt><dd><p>operations, respectively. Otherwise it’s done after. Default: <code class="docutils literal notranslate"><span class="pre">False</span></code> (after).</p>\n</dd>\n</dl>\n</dd>\n<dt>Examples::</dt><dd><div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">encoder_layer</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">TransformerEncoderLayer</span><span class="p">(</span><span class="n">d_model</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span> <span class="n">nhead</span><span class="o">=</span><span class="mi">8</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">src</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">rand</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">512</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">out</span> <span class="o">=</span> <span class="n">encoder_layer</span><span class="p">(</span><span class="n">src</span><span class="p">)</span>\n</pre></div>\n</div>\n</dd>\n<dt>Alternatively, when <code class="docutils literal notranslate"><span class="pre">batch_first</span></code> is <code class="docutils literal notranslate"><span class="pre">True</span></code>:</dt><dd><div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">encoder_layer</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">TransformerEncoderLayer</span><span class="p">(</span><span class="n">d_model</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span> <span class="n">nhead</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span> <span class="n">batch_first</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">src</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">rand</span><span class="p">(</span><span class="mi">32</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">512</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">out</span> <span class="o">=</span> <span class="n">encoder_layer</span><span class="p">(</span><span class="n">src</span><span class="p">)</span>\n</pre></div>\n</div>\n</dd>\n<dt>Fast path:</dt><dd><p>forward() will use a special optimized implementation described in\n<a class="reference external" href="https://arxiv.org/abs/2205.14135">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</a> if all of the following\nconditions are met:</p>\n<ul class="simple">\n<li><p>Either autograd is disabled (using <code class="docutils literal notranslate"><span class="pre">torch.inference_mode</span></code> or <code class="docutils literal notranslate"><span class="pre">torch.no_grad</span></code>) or no tensor\nargument <code class="docutils literal notranslate"><span class="pre">requires_grad</span></code></p></li>\n<li><p>training is disabled (using <code class="docutils literal notranslate"><span class="pre">.eval()</span></code>)</p></li>\n<li><p>batch_first is <code class="docutils literal notranslate"><span class="pre">True</span></code> and the input is batched (i.e., <code class="docutils literal notranslate"><span class="pre">src.dim()</span> <span class="pre">==</span> <span class="pre">3</span></code>)</p></li>\n<li><p>activation is one of: <code class="docutils literal notranslate"><span class="pre">&quot;relu&quot;</span></code>, <code class="docutils literal notranslate"><span class="pre">&quot;gelu&quot;</span></code>, <code class="docutils literal notranslate"><span class="pre">torch.functional.relu</span></code>, or <code class="docutils literal notranslate"><span class="pre">torch.functional.gelu</span></code></p></li>\n<li><p>at most one of <code class="docutils literal notranslate"><span class="pre">src_mask</span></code> and <code class="docutils literal notranslate"><span class="pre">src_key_padding_mask</span></code> is passed</p></li>\n<li><p>if src is a <a class="reference external" href="https://pytorch.org/docs/stable/nested.html">NestedTensor</a>, neither <code class="docutils literal notranslate"><span class="pre">src_mask</span></code>\nnor <code class="docutils literal notranslate"><span class="pre">src_key_padding_mask</span></code> is passed</p></li>\n<li><p>the two <code class="docutils literal notranslate"><span class="pre">LayerNorm</span></code> instances have a consistent <code class="docutils literal notranslate"><span class="pre">eps</span></code> value (this will naturally be the case\nunless the caller has manually modified one without modifying the other)</p></li>\n</ul>\n<p>If the optimized implementation is in use, a\n<a class="reference external" href="https://pytorch.org/docs/stable/nested.html">NestedTensor</a> can be\npassed for <code class="docutils literal notranslate"><span class="pre">src</span></code> to represent padding more efficiently than using a padding\nmask. In this case, a <a class="reference external" href="https://pytorch.org/docs/stable/nested.html">NestedTensor</a> will be\nreturned, and an additional speedup proportional to the fraction of the input that\nis padding can be expected.</p>\n</dd>\n</dl>\n</dd>\n</dl>\n\n\n</div>',
    outputType: "<class 'torch.Tensor'>",
    defaultArgs: {
      dim_feedforward: 2048,
      dropout: 0.1,
      activation: 'relu',
      layer_norm_eps: 0.00001,
      batch_first: false,
      norm_first: false,
    },
  },
  {
    classPath: 'fleet.model_builder.featurizers.MoleculeFeaturizer',
    component: {
      type: 'fleet.model_builder.featurizers.MoleculeFeaturizer',
      constructorArgsSummary: {
        allow_unknown: "<class 'bool'>",
        sym_bond_list: "<class 'bool'>",
        per_atom_fragmentation: "<class 'bool'>",
      },
      forwardArgsSummary: {
        mol: 'typing.Union[rdkit.Chem.rdchem.Mol, str, numpy.ndarray]',
      },
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>Small molecule featurizer.\nArgs:</p>\n<blockquote>\n<div><dl class="simple">\n<dt>allow_unknown (bool, optional): Boolean indicating whether to add an additional feature for out-of-vocabulary one-hot encoded</dt><dd><p>features.</p>\n</dd>\n</dl>\n<p>sym_bond_list (bool, optional): Boolean indicating whether bond list should be stored symmetrically.\nper_atom_fragmentation (bool, optional): if <cite>per_atom_fragmentation=true</cite> then multiple fragments of the molecule</p>\n<blockquote>\n<div><p>will be generated, the atoms will be removed one at a time resulting in a batch of fragments.</p>\n</div></blockquote>\n</div></blockquote>\n</div></blockquote>\n\n\n</div>',
    outputType: "<class 'torch_geometric.data.data.Data'>",
    defaultArgs: {},
  },
  {
    classPath: 'fleet.model_builder.featurizers.IntegerFeaturizer',
    component: {
      type: 'fleet.model_builder.featurizers.IntegerFeaturizer',
      constructorArgsSummary: {},
      forwardArgsSummary: {
        input_: "<class 'str'>",
      },
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>The integer featurizer</p>\n<p>Featurizes categorical data type columns to scalar tensors with dtype long</p>\n</div></blockquote>\n\n\n</div>',
    outputType: "<class 'numpy.ndarray'>",
    defaultArgs: {},
  },
  {
    classPath: 'fleet.model_builder.featurizers.DNASequenceFeaturizer',
    component: {
      type: 'fleet.model_builder.featurizers.DNASequenceFeaturizer',
      constructorArgsSummary: {},
      forwardArgsSummary: {
        input_: "<class 'str'>",
      },
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>DNA sequence featurizer</p>\n\n\n</div>',
    outputType: "<class 'numpy.ndarray'>",
    defaultArgs: {},
  },
  {
    classPath: 'fleet.model_builder.featurizers.RNASequenceFeaturizer',
    component: {
      type: 'fleet.model_builder.featurizers.RNASequenceFeaturizer',
      constructorArgsSummary: {},
      forwardArgsSummary: {
        input_: "<class 'str'>",
      },
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>RNA sequence featurizer</p>\n\n\n</div>',
    outputType: "<class 'numpy.ndarray'>",
    defaultArgs: {},
  },
  {
    classPath: 'fleet.model_builder.featurizers.ProteinSequenceFeaturizer',
    component: {
      type: 'fleet.model_builder.featurizers.ProteinSequenceFeaturizer',
      constructorArgsSummary: {},
      forwardArgsSummary: {
        input_: "<class 'str'>",
      },
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>Protein sequence featurizer</p>\n\n\n</div>',
    outputType: "<class 'numpy.ndarray'>",
    defaultArgs: {},
  },
  {
    classPath: 'molfeat.trans.fp.FPVecFilteredTransformer',
    component: {
      constructorArgsSummary: {
        length: "<class 'int'>?",
        del_invariant: "<class 'bool'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>Fingerprint molecule transformer with columns filters applying to the featurized vector when <cite>fit</cite> is called</p>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      del_invariant: false,
      length: 2000,
    },
  },
  {
    classPath: 'sklearn.preprocessing.LabelEncoder',
    component: {
      constructorArgsSummary: {},
      forwardArgsSummary: {},
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>Encode target labels with value between 0 and n_classes-1.</p>\n<blockquote>\n<div><p>This transformer should be used to encode target values, <em>i.e.</em> <cite>y</cite>, and\nnot the input <cite>X</cite>.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.12.</span></p>\n</div>\n<dl class="simple">\n<dt><a href="#id1"><span class="problematic" id="id2">classes_</span></a><span class="classifier">ndarray of shape (n_classes,)</span></dt><dd><p>Holds the label for each class.</p>\n</dd>\n</dl>\n<dl class="simple">\n<dt>OrdinalEncoder<span class="classifier">Encode categorical features using an ordinal encoding</span></dt><dd><p>scheme.</p>\n</dd>\n</dl>\n<p>OneHotEncoder : Encode categorical features as a one-hot numeric array.</p>\n<p><cite>LabelEncoder</cite> can be used to normalize labels.</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">preprocessing</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">le</span> <span class="o">=</span> <span class="n">preprocessing</span><span class="o">.</span><span class="n">LabelEncoder</span><span class="p">()</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">le</span><span class="o">.</span><span class="n">fit</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">6</span><span class="p">])</span>\n<span class="go">LabelEncoder()</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">le</span><span class="o">.</span><span class="n">classes_</span>\n<span class="go">array([1, 2, 6])</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">le</span><span class="o">.</span><span class="n">transform</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">6</span><span class="p">])</span>\n<span class="go">array([0, 0, 1, 2]...)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">le</span><span class="o">.</span><span class="n">inverse_transform</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">])</span>\n<span class="go">array([1, 1, 2, 6])</span>\n</pre></div>\n</div>\n<p>It can also be used to transform non-numerical labels (as long as they are\nhashable and comparable) to numerical labels.</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">le</span> <span class="o">=</span> <span class="n">preprocessing</span><span class="o">.</span><span class="n">LabelEncoder</span><span class="p">()</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">le</span><span class="o">.</span><span class="n">fit</span><span class="p">([</span><span class="s2">&quot;paris&quot;</span><span class="p">,</span> <span class="s2">&quot;paris&quot;</span><span class="p">,</span> <span class="s2">&quot;tokyo&quot;</span><span class="p">,</span> <span class="s2">&quot;amsterdam&quot;</span><span class="p">])</span>\n<span class="go">LabelEncoder()</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">list</span><span class="p">(</span><span class="n">le</span><span class="o">.</span><span class="n">classes_</span><span class="p">)</span>\n<span class="go">[&#39;amsterdam&#39;, &#39;paris&#39;, &#39;tokyo&#39;]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">le</span><span class="o">.</span><span class="n">transform</span><span class="p">([</span><span class="s2">&quot;tokyo&quot;</span><span class="p">,</span> <span class="s2">&quot;tokyo&quot;</span><span class="p">,</span> <span class="s2">&quot;paris&quot;</span><span class="p">])</span>\n<span class="go">array([2, 2, 1]...)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">list</span><span class="p">(</span><span class="n">le</span><span class="o">.</span><span class="n">inverse_transform</span><span class="p">([</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">]))</span>\n<span class="go">[&#39;tokyo&#39;, &#39;tokyo&#39;, &#39;paris&#39;]</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {},
  },
  {
    classPath: 'sklearn.preprocessing.OneHotEncoder',
    component: {
      constructorArgsSummary: {},
      forwardArgsSummary: {},
    },
    type: 'featurizer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>Encode categorical features as a one-hot numeric array.</p>\n<p>The input to this transformer should be an array-like of integers or\nstrings, denoting the values taken on by categorical (discrete) features.\nThe features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’)\nencoding scheme. This creates a binary column for each category and\nreturns a sparse matrix or dense array (depending on the <code class="docutils literal notranslate"><span class="pre">sparse_output</span></code>\nparameter)</p>\n<p>By default, the encoder derives the categories based on the unique values\nin each feature. Alternatively, you can also specify the <cite>categories</cite>\nmanually.</p>\n<p>This encoding is needed for feeding categorical data to many scikit-learn\nestimators, notably linear models and SVMs with the standard kernels.</p>\n<p>Note: a one-hot encoding of y labels should use a LabelBinarizer\ninstead.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<dl>\n<dt>categories<span class="classifier">‘auto’ or a list of array-like, default=’auto’</span></dt><dd><p>Categories (unique values) per feature:</p>\n<ul class="simple">\n<li><p>‘auto’ : Determine categories automatically from the training data.</p></li>\n<li><p>list : <code class="docutils literal notranslate"><span class="pre">categories[i]</span></code> holds the categories expected in the ith\ncolumn. The passed categories should not mix strings and numeric\nvalues within a single feature, and should be sorted in case of\nnumeric values.</p></li>\n</ul>\n<p>The used categories can be found in the <code class="docutils literal notranslate"><span class="pre">categories_</span></code> attribute.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.20.</span></p>\n</div>\n</dd>\n<dt>drop<span class="classifier">{‘first’, ‘if_binary’} or an array-like of shape (n_features,),             default=None</span></dt><dd><p>Specifies a methodology to use to drop one of the categories per\nfeature. This is useful in situations where perfectly collinear\nfeatures cause problems, such as when feeding the resulting data\ninto an unregularized linear regression model.</p>\n<p>However, dropping one category breaks the symmetry of the original\nrepresentation and can therefore induce a bias in downstream models,\nfor instance for penalized linear classification or regression models.</p>\n<ul class="simple">\n<li><p>None : retain all features (the default).</p></li>\n<li><p>‘first’ : drop the first category in each feature. If only one\ncategory is present, the feature will be dropped entirely.</p></li>\n<li><p>‘if_binary’ : drop the first category in each feature with two\ncategories. Features with 1 or more than 2 categories are\nleft intact.</p></li>\n<li><p>array : <code class="docutils literal notranslate"><span class="pre">drop[i]</span></code> is the category in feature <code class="docutils literal notranslate"><span class="pre">X[:,</span> <span class="pre">i]</span></code> that\nshould be dropped.</p></li>\n</ul>\n<p>When <cite>max_categories</cite> or <cite>min_frequency</cite> is configured to group\ninfrequent categories, the dropping behavior is handled after the\ngrouping.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.21: </span>The parameter <cite>drop</cite> was added in 0.21.</p>\n</div>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.23: </span>The option <cite>drop=’if_binary’</cite> was added in 0.23.</p>\n</div>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 1.1: </span>Support for dropping infrequent categories.</p>\n</div>\n</dd>\n<dt>sparse<span class="classifier">bool, default=True</span></dt><dd><p>Will return sparse matrix if set True else will return an array.</p>\n<div class="deprecated">\n<p><span class="versionmodified deprecated">Deprecated since version 1.2: </span><cite>sparse</cite> is deprecated in 1.2 and will be removed in 1.4. Use\n<cite>sparse_output</cite> instead.</p>\n</div>\n</dd>\n<dt>sparse_output<span class="classifier">bool, default=True</span></dt><dd><p>Will return sparse matrix if set True else will return an array.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.2: </span><cite>sparse</cite> was renamed to <cite>sparse_output</cite></p>\n</div>\n</dd>\n<dt>dtype<span class="classifier">number type, default=float</span></dt><dd><p>Desired dtype of output.</p>\n</dd>\n<dt>handle_unknown<span class="classifier">{‘error’, ‘ignore’, ‘infrequent_if_exist’},                      default=’error’</span></dt><dd><p>Specifies the way unknown categories are handled during <code class="xref py py-meth docutils literal notranslate"><span class="pre">transform()</span></code>.</p>\n<ul class="simple">\n<li><p>‘error’ : Raise an error if an unknown category is present during transform.</p></li>\n<li><p>‘ignore’ : When an unknown category is encountered during\ntransform, the resulting one-hot encoded columns for this feature\nwill be all zeros. In the inverse transform, an unknown category\nwill be denoted as None.</p></li>\n<li><p>‘infrequent_if_exist’ : When an unknown category is encountered\nduring transform, the resulting one-hot encoded columns for this\nfeature will map to the infrequent category if it exists. The\ninfrequent category will be mapped to the last position in the\nencoding. During inverse transform, an unknown category will be\nmapped to the category denoted <cite>‘infrequent’</cite> if it exists. If the\n<cite>‘infrequent’</cite> category does not exist, then <code class="xref py py-meth docutils literal notranslate"><span class="pre">transform()</span></code> and\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">inverse_transform()</span></code> will handle an unknown category as with\n<cite>handle_unknown=’ignore’</cite>. Infrequent categories exist based on\n<cite>min_frequency</cite> and <cite>max_categories</cite>. Read more in the\n<span class="xref std std-ref">User Guide</span>.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 1.1: </span><cite>‘infrequent_if_exist’</cite> was added to automatically handle unknown\ncategories and infrequent categories.</p>\n</div>\n</dd>\n<dt>min_frequency<span class="classifier">int or float, default=None</span></dt><dd><p>Specifies the minimum frequency below which a category will be\nconsidered infrequent.</p>\n<ul class="simple">\n<li><p>If <cite>int</cite>, categories with a smaller cardinality will be considered\ninfrequent.</p></li>\n<li><p>If <cite>float</cite>, categories with a smaller cardinality than\n<cite>min_frequency * n_samples</cite>  will be considered infrequent.</p></li>\n</ul>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.1: </span>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n</div>\n</dd>\n<dt>max_categories<span class="classifier">int, default=None</span></dt><dd><p>Specifies an upper limit to the number of output features for each input\nfeature when considering infrequent categories. If there are infrequent\ncategories, <cite>max_categories</cite> includes the category representing the\ninfrequent categories along with the frequent categories. If <cite>None</cite>,\nthere is no limit to the number of output features.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.1: </span>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n</div>\n</dd>\n<dt>feature_name_combiner<span class="classifier">“concat” or callable, default=”concat”</span></dt><dd><p>Callable with signature <cite>def callable(input_feature, category)</cite> that returns a\nstring. This is used to create feature names to be returned by\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">get_feature_names_out()</span></code>.</p>\n<p><cite>“concat”</cite> concatenates encoded feature name and category with\n<cite>feature + “_” + str(category)</cite>.E.g. feature X with values 1, 6, 7 create\nfeature names <cite>X_1, X_6, X_7</cite>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.3.</span></p>\n</div>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id1"><span class="problematic" id="id2">categories_</span></a><span class="classifier">list of arrays</span></dt><dd><p>The categories of each feature determined during fitting\n(in order of the features in X and corresponding with the output\nof <code class="docutils literal notranslate"><span class="pre">transform</span></code>). This includes the category specified in <code class="docutils literal notranslate"><span class="pre">drop</span></code>\n(if any).</p>\n</dd>\n<dt><a href="#id3"><span class="problematic" id="id4">drop_idx_</span></a><span class="classifier">array of shape (n_features,)</span></dt><dd><ul class="simple">\n<li><p><code class="docutils literal notranslate"><span class="pre">drop_idx_[i]</span></code> is the index in <code class="docutils literal notranslate"><span class="pre">categories_[i]</span></code> of the category\nto be dropped for each feature.</p></li>\n<li><p><code class="docutils literal notranslate"><span class="pre">drop_idx_[i]</span> <span class="pre">=</span> <span class="pre">None</span></code> if no category is to be dropped from the\nfeature with index <code class="docutils literal notranslate"><span class="pre">i</span></code>, e.g. when <cite>drop=’if_binary’</cite> and the\nfeature isn’t binary.</p></li>\n<li><p><code class="docutils literal notranslate"><span class="pre">drop_idx_</span> <span class="pre">=</span> <span class="pre">None</span></code> if all the transformed features will be\nretained.</p></li>\n</ul>\n<p>If infrequent categories are enabled by setting <cite>min_frequency</cite> or\n<cite>max_categories</cite> to a non-default value and <cite>drop_idx[i]</cite> corresponds\nto a infrequent category, then the entire infrequent category is\ndropped.</p>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.23: </span>Added the possibility to contain <cite>None</cite> values.</p>\n</div>\n</dd>\n<dt><a href="#id5"><span class="problematic" id="id6">infrequent_categories_</span></a><span class="classifier">list of ndarray</span></dt><dd><p>Defined only if infrequent categories are enabled by setting\n<cite>min_frequency</cite> or <cite>max_categories</cite> to a non-default value.\n<cite>infrequent_categories_[i]</cite> are the infrequent categories for feature\n<cite>i</cite>. If the feature <cite>i</cite> has no infrequent categories\n<cite>infrequent_categories_[i]</cite> is None.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.1.</span></p>\n</div>\n</dd>\n<dt><a href="#id7"><span class="problematic" id="id8">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id9"><span class="problematic" id="id10">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt>feature_name_combiner<span class="classifier">callable or None</span></dt><dd><p>Callable with signature <cite>def callable(input_feature, category)</cite> that returns a\nstring. This is used to create feature names to be returned by\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">get_feature_names_out()</span></code>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.3.</span></p>\n</div>\n</dd>\n</dl>\n<dl class="simple">\n<dt>OrdinalEncoder<span class="classifier">Performs an ordinal (integer)</span></dt><dd><p>encoding of the categorical features.</p>\n</dd>\n</dl>\n<p>TargetEncoder : Encodes categorical features using the target.\nsklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of</p>\n<blockquote>\n<div><p>dictionary items (also handles string-valued features).</p>\n</div></blockquote>\n<dl class="simple">\n<dt>sklearn.feature_extraction.FeatureHasher<span class="classifier">Performs an approximate one-hot</span></dt><dd><p>encoding of dictionary items or strings.</p>\n</dd>\n<dt>LabelBinarizer<span class="classifier">Binarizes labels in a one-vs-all</span></dt><dd><p>fashion.</p>\n</dd>\n<dt>MultiLabelBinarizer<span class="classifier">Transforms between iterable of</span></dt><dd><p>iterables and a multilabel format, e.g. a (samples x classes) binary\nmatrix indicating the presence of a class label.</p>\n</dd>\n</dl>\n<p>Given a dataset with two features, we let the encoder find the unique\nvalues per feature and transform the data to a binary one-hot encoding.</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="kn">import</span> <span class="n">OneHotEncoder</span>\n</pre></div>\n</div>\n<p>One can discard categories not seen during <cite>fit</cite>:</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">enc</span> <span class="o">=</span> <span class="n">OneHotEncoder</span><span class="p">(</span><span class="n">handle_unknown</span><span class="o">=</span><span class="s1">&#39;ignore&#39;</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">X</span> <span class="o">=</span> <span class="p">[[</span><span class="s1">&#39;Male&#39;</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="s1">&#39;Female&#39;</span><span class="p">,</span> <span class="mi">3</span><span class="p">],</span> <span class="p">[</span><span class="s1">&#39;Female&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">]]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">enc</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>\n<span class="go">OneHotEncoder(handle_unknown=&#39;ignore&#39;)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">enc</span><span class="o">.</span><span class="n">categories_</span>\n<span class="go">[array([&#39;Female&#39;, &#39;Male&#39;], dtype=object), array([1, 2, 3], dtype=object)]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">enc</span><span class="o">.</span><span class="n">transform</span><span class="p">([[</span><span class="s1">&#39;Female&#39;</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="s1">&#39;Male&#39;</span><span class="p">,</span> <span class="mi">4</span><span class="p">]])</span><span class="o">.</span><span class="n">toarray</span><span class="p">()</span>\n<span class="go">array([[1., 0., 1., 0., 0.],</span>\n<span class="go">       [0., 1., 0., 0., 0.]])</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">enc</span><span class="o">.</span><span class="n">inverse_transform</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">]])</span>\n<span class="go">array([[&#39;Male&#39;, 1],</span>\n<span class="go">       [None, 2]], dtype=object)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">enc</span><span class="o">.</span><span class="n">get_feature_names_out</span><span class="p">([</span><span class="s1">&#39;gender&#39;</span><span class="p">,</span> <span class="s1">&#39;group&#39;</span><span class="p">])</span>\n<span class="go">array([&#39;gender_Female&#39;, &#39;gender_Male&#39;, &#39;group_1&#39;, &#39;group_2&#39;, &#39;group_3&#39;], ...)</span>\n</pre></div>\n</div>\n<p>One can always drop the first column for each feature:</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">drop_enc</span> <span class="o">=</span> <span class="n">OneHotEncoder</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="s1">&#39;first&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">drop_enc</span><span class="o">.</span><span class="n">categories_</span>\n<span class="go">[array([&#39;Female&#39;, &#39;Male&#39;], dtype=object), array([1, 2, 3], dtype=object)]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">drop_enc</span><span class="o">.</span><span class="n">transform</span><span class="p">([[</span><span class="s1">&#39;Female&#39;</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="s1">&#39;Male&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">]])</span><span class="o">.</span><span class="n">toarray</span><span class="p">()</span>\n<span class="go">array([[0., 0., 0.],</span>\n<span class="go">       [1., 1., 0.]])</span>\n</pre></div>\n</div>\n<p>Or drop a column for feature only having 2 categories:</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">drop_binary_enc</span> <span class="o">=</span> <span class="n">OneHotEncoder</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="s1">&#39;if_binary&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">drop_binary_enc</span><span class="o">.</span><span class="n">transform</span><span class="p">([[</span><span class="s1">&#39;Female&#39;</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="s1">&#39;Male&#39;</span><span class="p">,</span> <span class="mi">2</span><span class="p">]])</span><span class="o">.</span><span class="n">toarray</span><span class="p">()</span>\n<span class="go">array([[0., 1., 0., 0.],</span>\n<span class="go">       [1., 0., 1., 0.]])</span>\n</pre></div>\n</div>\n<p>One can change the way feature names are created.</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="k">def</span> <span class="nf">custom_combiner</span><span class="p">(</span><span class="n">feature</span><span class="p">,</span> <span class="n">category</span><span class="p">):</span>\n<span class="gp">... </span>    <span class="k">return</span> <span class="nb">str</span><span class="p">(</span><span class="n">feature</span><span class="p">)</span> <span class="o">+</span> <span class="s2">&quot;_&quot;</span> <span class="o">+</span> <span class="nb">type</span><span class="p">(</span><span class="n">category</span><span class="p">)</span><span class="o">.</span><span class="vm">__name__</span> <span class="o">+</span> <span class="s2">&quot;_&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">category</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">custom_fnames_enc</span> <span class="o">=</span> <span class="n">OneHotEncoder</span><span class="p">(</span><span class="n">feature_name_combiner</span><span class="o">=</span><span class="n">custom_combiner</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">custom_fnames_enc</span><span class="o">.</span><span class="n">get_feature_names_out</span><span class="p">()</span>\n<span class="go">array([&#39;x0_str_Female&#39;, &#39;x0_str_Male&#39;, &#39;x1_int_1&#39;, &#39;x1_int_2&#39;, &#39;x1_int_3&#39;],</span>\n<span class="go">      dtype=object)</span>\n</pre></div>\n</div>\n<p>Infrequent categories are enabled by setting <cite>max_categories</cite> or <cite>min_frequency</cite>.</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="s2">&quot;a&quot;</span><span class="p">]</span> <span class="o">*</span> <span class="mi">5</span> <span class="o">+</span> <span class="p">[</span><span class="s2">&quot;b&quot;</span><span class="p">]</span> <span class="o">*</span> <span class="mi">20</span> <span class="o">+</span> <span class="p">[</span><span class="s2">&quot;c&quot;</span><span class="p">]</span> <span class="o">*</span> <span class="mi">10</span> <span class="o">+</span> <span class="p">[</span><span class="s2">&quot;d&quot;</span><span class="p">]</span> <span class="o">*</span> <span class="mi">3</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="nb">object</span><span class="p">)</span><span class="o">.</span><span class="n">T</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">ohe</span> <span class="o">=</span> <span class="n">OneHotEncoder</span><span class="p">(</span><span class="n">max_categories</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">sparse_output</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">ohe</span><span class="o">.</span><span class="n">infrequent_categories_</span>\n<span class="go">[array([&#39;a&#39;, &#39;d&#39;], dtype=object)]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">ohe</span><span class="o">.</span><span class="n">transform</span><span class="p">([[</span><span class="s2">&quot;a&quot;</span><span class="p">],</span> <span class="p">[</span><span class="s2">&quot;b&quot;</span><span class="p">]])</span>\n<span class="go">array([[0., 0., 1.],</span>\n<span class="go">       [1., 0., 0.]])</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {},
  },
  {
    classPath: 'sklearn.preprocessing.StandardScaler',
    component: {
      constructorArgsSummary: {
        with_mean: "<class 'bool'>?",
        with_std: "<class 'bool'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'transformer',
    argsOptions: null,
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>Standardize features by removing the mean and scaling to unit variance.</p>\n<blockquote>\n<div><p>The standard score of a sample <cite>x</cite> is calculated as:</p>\n<blockquote>\n<div><p>z = (x - u) / s</p>\n</div></blockquote>\n<p>where <cite>u</cite> is the mean of the training samples or zero if <cite>with_mean=False</cite>,\nand <cite>s</cite> is the standard deviation of the training samples or one if\n<cite>with_std=False</cite>.</p>\n<p>Centering and scaling happen independently on each feature by computing\nthe relevant statistics on the samples in the training set. Mean and\nstandard deviation are then stored to be used on later data using\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">transform()</span></code>.</p>\n<p>Standardization of a dataset is a common requirement for many\nmachine learning estimators: they might behave badly if the\nindividual features do not more or less look like standard normally\ndistributed data (e.g. Gaussian with 0 mean and unit variance).</p>\n<p>For instance many elements used in the objective function of\na learning algorithm (such as the RBF kernel of Support Vector\nMachines or the L1 and L2 regularizers of linear models) assume that\nall features are centered around 0 and have variance in the same\norder. If a feature has a variance that is orders of magnitude larger\nthan others, it might dominate the objective function and make the\nestimator unable to learn from other features correctly as expected.</p>\n<p>This scaler can also be applied to sparse CSR or CSC matrices by passing\n<cite>with_mean=False</cite> to avoid breaking the sparsity structure of the data.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<dl class="simple">\n<dt>copy<span class="classifier">bool, default=True</span></dt><dd><p>If False, try to avoid a copy and do inplace scaling instead.\nThis is not guaranteed to always work inplace; e.g. if the data is\nnot a NumPy array or scipy.sparse CSR matrix, a copy may still be\nreturned.</p>\n</dd>\n<dt>with_mean<span class="classifier">bool, default=True</span></dt><dd><p>If True, center the data before scaling.\nThis does not work (and will raise an exception) when attempted on\nsparse matrices, because centering them entails building a dense\nmatrix which in common use cases is likely to be too large to fit in\nmemory.</p>\n</dd>\n<dt>with_std<span class="classifier">bool, default=True</span></dt><dd><p>If True, scale the data to unit variance (or equivalently,\nunit standard deviation).</p>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id1"><span class="problematic" id="id2">scale_</span></a><span class="classifier">ndarray of shape (n_features,) or None</span></dt><dd><p>Per feature relative scaling of the data to achieve zero mean and unit\nvariance. Generally this is calculated using <cite>np.sqrt(var_)</cite>. If a\nvariance is zero, we can’t achieve unit variance, and the data is left\nas-is, giving a scaling factor of 1. <cite>scale_</cite> is equal to <cite>None</cite>\nwhen <cite>with_std=False</cite>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.17: </span><em>scale_</em></p>\n</div>\n</dd>\n<dt><a href="#id3"><span class="problematic" id="id4">mean_</span></a><span class="classifier">ndarray of shape (n_features,) or None</span></dt><dd><p>The mean value for each feature in the training set.\nEqual to <code class="docutils literal notranslate"><span class="pre">None</span></code> when <code class="docutils literal notranslate"><span class="pre">with_mean=False</span></code>.</p>\n</dd>\n<dt><a href="#id5"><span class="problematic" id="id6">var_</span></a><span class="classifier">ndarray of shape (n_features,) or None</span></dt><dd><p>The variance for each feature in the training set. Used to compute\n<cite>scale_</cite>. Equal to <code class="docutils literal notranslate"><span class="pre">None</span></code> when <code class="docutils literal notranslate"><span class="pre">with_std=False</span></code>.</p>\n</dd>\n<dt><a href="#id7"><span class="problematic" id="id8">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.24.</span></p>\n</div>\n</dd>\n<dt><a href="#id9"><span class="problematic" id="id10">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id11"><span class="problematic" id="id12">n_samples_seen_</span></a><span class="classifier">int or ndarray of shape (n_features,)</span></dt><dd><p>The number of samples processed by the estimator for each feature.\nIf there are no missing samples, the <code class="docutils literal notranslate"><span class="pre">n_samples_seen</span></code> will be an\ninteger, otherwise it will be an array of dtype int. If\n<cite>sample_weights</cite> are used it will be a float (if no missing data)\nor an array of dtype float that sums the weights seen so far.\nWill be reset on new calls to fit, but increments across\n<code class="docutils literal notranslate"><span class="pre">partial_fit</span></code> calls.</p>\n</dd>\n</dl>\n<p>scale : Equivalent function without the estimator API.</p>\n<dl class="simple">\n<dt><code class="xref py py-class docutils literal notranslate"><span class="pre">PCA</span></code><span class="classifier">Further removes the linear</span></dt><dd><p>correlation across features with ‘whiten=True’.</p>\n</dd>\n</dl>\n<p>NaNs are treated as missing values: disregarded in fit, and maintained in\ntransform.</p>\n<p>We use a biased estimator for the standard deviation, equivalent to\n<cite>numpy.std(x, ddof=0)</cite>. Note that the choice of <cite>ddof</cite> is unlikely to\naffect model performance.</p>\n<p>For a comparison of the different scalers, transformers, and normalizers,\nsee <span class="xref std std-ref">examples/preprocessing/plot_all_scaling.py</span>.</p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="kn">import</span> <span class="n">StandardScaler</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">data</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">scaler</span> <span class="o">=</span> <span class="n">StandardScaler</span><span class="p">()</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">scaler</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">data</span><span class="p">))</span>\n<span class="go">StandardScaler()</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">scaler</span><span class="o">.</span><span class="n">mean_</span><span class="p">)</span>\n<span class="go">[0.5 0.5]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">scaler</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">data</span><span class="p">))</span>\n<span class="go">[[-1. -1.]</span>\n<span class="go"> [-1. -1.]</span>\n<span class="go"> [ 1.  1.]</span>\n<span class="go"> [ 1.  1.]]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">scaler</span><span class="o">.</span><span class="n">transform</span><span class="p">([[</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">]]))</span>\n<span class="go">[[3. 3.]]</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      with_mean: true,
      with_std: true,
    },
  },
  {
    classPath: 'sklearn.neighbors.KNeighborsRegressor',
    component: {
      constructorArgsSummary: {
        n_neighbors: "<class 'int'>?",
        algorithm: "<class 'str'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'scikit_reg',
    argsOptions: {
      algorithm: [
        {
          key: 'auto',
          label: null,
          latex: null,
        },
        {
          key: 'ball_tree',
          label: null,
          latex: null,
        },
        {
          key: 'kd_tree',
          label: null,
          latex: null,
        },
        {
          key: 'brute',
          label: null,
          latex: null,
        },
      ],
      metric: [
        {
          key: 'l1',
          label: 'cityblock (l1)',
          latex: '...',
        },
        {
          key: 'l2',
          label: 'euclidean (l2)',
          latex: '...',
        },
        {
          key: 'haversine',
          label: 'haversine',
          latex: '...',
        },
        {
          key: 'cosine',
          label: 'cosine',
          latex: '...',
        },
      ],
    },
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>Regression based on k-nearest neighbors.</p>\n<blockquote>\n<div><p>The target is predicted by local interpolation of the targets\nassociated of the nearest neighbors in the training set.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.9.</span></p>\n</div>\n<dl>\n<dt>n_neighbors<span class="classifier">int, default=5</span></dt><dd><p>Number of neighbors to use by default for <code class="xref py py-meth docutils literal notranslate"><span class="pre">kneighbors()</span></code> queries.</p>\n</dd>\n<dt>weights<span class="classifier">{‘uniform’, ‘distance’}, callable or None, default=’uniform’</span></dt><dd><p>Weight function used in prediction.  Possible values:</p>\n<ul class="simple">\n<li><p>‘uniform’ : uniform weights.  All points in each neighborhood\nare weighted equally.</p></li>\n<li><p>‘distance’ : weight points by the inverse of their distance.\nin this case, closer neighbors of a query point will have a\ngreater influence than neighbors which are further away.</p></li>\n<li><p>[callable] : a user-defined function which accepts an\narray of distances, and returns an array of the same shape\ncontaining the weights.</p></li>\n</ul>\n<p>Uniform weights are used by default.</p>\n</dd>\n<dt>algorithm<span class="classifier">{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’</span></dt><dd><p>Algorithm used to compute the nearest neighbors:</p>\n<ul class="simple">\n<li><p>‘ball_tree’ will use <code class="xref py py-class docutils literal notranslate"><span class="pre">BallTree</span></code></p></li>\n<li><p>‘kd_tree’ will use <code class="xref py py-class docutils literal notranslate"><span class="pre">KDTree</span></code></p></li>\n<li><p>‘brute’ will use a brute-force search.</p></li>\n<li><p>‘auto’ will attempt to decide the most appropriate algorithm\nbased on the values passed to <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code> method.</p></li>\n</ul>\n<p>Note: fitting on sparse input will override the setting of\nthis parameter, using brute force.</p>\n</dd>\n<dt>leaf_size<span class="classifier">int, default=30</span></dt><dd><p>Leaf size passed to BallTree or KDTree.  This can affect the\nspeed of the construction and query, as well as the memory\nrequired to store the tree.  The optimal value depends on the\nnature of the problem.</p>\n</dd>\n<dt>p<span class="classifier">int, default=2</span></dt><dd><p>Power parameter for the Minkowski metric. When p = 1, this is\nequivalent to using manhattan_distance (l1), and euclidean_distance\n(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.</p>\n</dd>\n<dt>metric<span class="classifier">str or callable, default=’minkowski’</span></dt><dd><p>Metric to use for distance computation. Default is “minkowski”, which\nresults in the standard Euclidean distance when p = 2. See the\ndocumentation of <a class="reference external" href="https://docs.scipy.org/doc/scipy/reference/spatial.distance.html">scipy.spatial.distance</a> and\nthe metrics listed in\n<code class="xref py py-class docutils literal notranslate"><span class="pre">distance_metrics</span></code> for valid metric\nvalues.</p>\n<p>If metric is “precomputed”, X is assumed to be a distance matrix and\nmust be square during fit. X may be a <span class="xref std std-term">sparse graph</span>, in which\ncase only “nonzero” elements may be considered neighbors.</p>\n<p>If metric is a callable function, it takes two arrays representing 1D\nvectors as inputs and must return one value indicating the distance\nbetween those vectors. This works for Scipy’s metrics, but is less\nefficient than passing the metric name as a string.</p>\n</dd>\n<dt>metric_params<span class="classifier">dict, default=None</span></dt><dd><p>Additional keyword arguments for the metric function.</p>\n</dd>\n<dt>n_jobs<span class="classifier">int, default=None</span></dt><dd><p>The number of parallel jobs to run for neighbors search.\n<code class="docutils literal notranslate"><span class="pre">None</span></code> means 1 unless in a <code class="xref py py-obj docutils literal notranslate"><span class="pre">joblib.parallel_backend</span></code> context.\n<code class="docutils literal notranslate"><span class="pre">-1</span></code> means using all processors. See <span class="xref std std-term">Glossary</span>\nfor more details.\nDoesn’t affect <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code> method.</p>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id1"><span class="problematic" id="id2">effective_metric_</span></a><span class="classifier">str or callable</span></dt><dd><p>The distance metric to use. It will be same as the <cite>metric</cite> parameter\nor a synonym of it, e.g. ‘euclidean’ if the <cite>metric</cite> parameter set to\n‘minkowski’ and <cite>p</cite> parameter set to 2.</p>\n</dd>\n<dt><a href="#id3"><span class="problematic" id="id4">effective_metric_params_</span></a><span class="classifier">dict</span></dt><dd><p>Additional keyword arguments for the metric function. For most metrics\nwill be same with <cite>metric_params</cite> parameter, but may also contain the\n<cite>p</cite> parameter value if the <cite>effective_metric_</cite> attribute is set to\n‘minkowski’.</p>\n</dd>\n<dt><a href="#id5"><span class="problematic" id="id6">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.24.</span></p>\n</div>\n</dd>\n<dt><a href="#id7"><span class="problematic" id="id8">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id9"><span class="problematic" id="id10">n_samples_fit_</span></a><span class="classifier">int</span></dt><dd><p>Number of samples in the fitted data.</p>\n</dd>\n</dl>\n<p>NearestNeighbors : Unsupervised learner for implementing neighbor searches.\nRadiusNeighborsRegressor : Regression based on neighbors within a fixed radius.\nKNeighborsClassifier : Classifier implementing the k-nearest neighbors vote.\nRadiusNeighborsClassifier : Classifier implementing</p>\n<blockquote>\n<div><p>a vote among neighbors within a given radius.</p>\n</div></blockquote>\n<p>See <span class="xref std std-ref">Nearest Neighbors</span> in the online documentation\nfor a discussion of the choice of <code class="docutils literal notranslate"><span class="pre">algorithm</span></code> and <code class="docutils literal notranslate"><span class="pre">leaf_size</span></code>.</p>\n<div class="admonition warning">\n<p class="admonition-title">Warning</p>\n<p>Regarding the Nearest Neighbors algorithms, if it is found that two\nneighbors, neighbor <cite>k+1</cite> and <cite>k</cite>, have identical distances but\ndifferent labels, the results will depend on the ordering of the\ntraining data.</p>\n</div>\n<p><a class="reference external" href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm</a></p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">X</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">2</span><span class="p">],</span> <span class="p">[</span><span class="mi">3</span><span class="p">]]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">y</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsRegressor</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">neigh</span> <span class="o">=</span> <span class="n">KNeighborsRegressor</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">neigh</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>\n<span class="go">KNeighborsRegressor(...)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">neigh</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mf">1.5</span><span class="p">]]))</span>\n<span class="go">[0.5]</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      n_neighbors: 5,
      algorithm: 'kd_tree',
    },
  },
  {
    classPath: 'sklearn.ensemble.RandomForestRegressor',
    component: {
      constructorArgsSummary: {
        n_estimators: "<class 'int'>?",
        criterion: "<class 'str'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'scikit_reg',
    argsOptions: {
      criterion: [
        {
          key: 'squared_error',
          label: 'Squared Error',
          latex: null,
        },
        {
          key: 'absolute_error',
          label: 'Absolute Error',
          latex: null,
        },
        {
          key: 'Friedman MSE',
          label: 'friedman_mse',
          latex: null,
        },
        {
          key: 'Poisson',
          label: 'poisson',
          latex: null,
        },
      ],
      max_features: [
        {
          key: 'sqrt',
          label: 'sqrt(n_features)',
          latex: '...',
        },
        {
          key: 'log2',
          label: 'log2(n_features)',
          latex: '...',
        },
      ],
    },
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>A random forest regressor.</p>\n<p>A random forest is a meta estimator that fits a number of classifying\ndecision trees on various sub-samples of the dataset and uses averaging\nto improve the predictive accuracy and control over-fitting.\nThe sub-sample size is controlled with the <cite>max_samples</cite> parameter if\n<cite>bootstrap=True</cite> (default), otherwise the whole dataset is used to build\neach tree.</p>\n<p>For a comparison between tree-based ensemble models see the example\n<span class="xref std std-ref">sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py</span>.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<dl>\n<dt>n_estimators<span class="classifier">int, default=100</span></dt><dd><p>The number of trees in the forest.</p>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.22: </span>The default value of <code class="docutils literal notranslate"><span class="pre">n_estimators</span></code> changed from 10 to 100\nin 0.22.</p>\n</div>\n</dd>\n<dt>criterion<span class="classifier">{“squared_error”, “absolute_error”, “friedman_mse”, “poisson”},             default=”squared_error”</span></dt><dd><p>The function to measure the quality of a split. Supported criteria\nare “squared_error” for the mean squared error, which is equal to\nvariance reduction as feature selection criterion and minimizes the L2\nloss using the mean of each terminal node, “friedman_mse”, which uses\nmean squared error with Friedman’s improvement score for potential\nsplits, “absolute_error” for the mean absolute error, which minimizes\nthe L1 loss using the median of each terminal node, and “poisson” which\nuses reduction in Poisson deviance to find splits.\nTraining using “absolute_error” is significantly slower\nthan when using “squared_error”.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.18: </span>Mean Absolute Error (MAE) criterion.</p>\n</div>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0: </span>Poisson criterion.</p>\n</div>\n</dd>\n<dt>max_depth<span class="classifier">int, default=None</span></dt><dd><p>The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.</p>\n</dd>\n<dt>min_samples_split<span class="classifier">int or float, default=2</span></dt><dd><p>The minimum number of samples required to split an internal node:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_split</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_split</cite> is a fraction and\n<cite>ceil(min_samples_split * n_samples)</cite> are the minimum\nnumber of samples for each split.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_samples_leaf<span class="classifier">int or float, default=1</span></dt><dd><p>The minimum number of samples required to be at a leaf node.\nA split point at any depth will only be considered if it leaves at\nleast <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code> training samples in each of the left and\nright branches.  This may have the effect of smoothing the model,\nespecially in regression.</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_leaf</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_leaf</cite> is a fraction and\n<cite>ceil(min_samples_leaf * n_samples)</cite> are the minimum\nnumber of samples for each node.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_weight_fraction_leaf<span class="classifier">float, default=0.0</span></dt><dd><p>The minimum weighted fraction of the sum total of weights (of all\nthe input samples) required to be at a leaf node. Samples have\nequal weight when sample_weight is not provided.</p>\n</dd>\n<dt>max_features<span class="classifier">{“sqrt”, “log2”, None}, int or float, default=1.0</span></dt><dd><p>The number of features to consider when looking for the best split:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>max_features</cite> features at each split.</p></li>\n<li><p>If float, then <cite>max_features</cite> is a fraction and\n<cite>max(1, int(max_features * n_features_in_))</cite> features are considered at each\nsplit.</p></li>\n<li><p>If “sqrt”, then <cite>max_features=sqrt(n_features)</cite>.</p></li>\n<li><p>If “log2”, then <cite>max_features=log2(n_features)</cite>.</p></li>\n<li><p>If None or 1.0, then <cite>max_features=n_features</cite>.</p></li>\n</ul>\n<div class="admonition note">\n<p class="admonition-title">Note</p>\n<p>The default of 1.0 is equivalent to bagged trees and more\nrandomness can be achieved by setting smaller values, e.g. 0.3.</p>\n</div>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 1.1: </span>The default of <cite>max_features</cite> changed from <cite>“auto”</cite> to 1.0.</p>\n</div>\n<p>Note: the search for a split does not stop until at least one\nvalid partition of the node samples is found, even if it requires to\neffectively inspect more than <code class="docutils literal notranslate"><span class="pre">max_features</span></code> features.</p>\n</dd>\n<dt>max_leaf_nodes<span class="classifier">int, default=None</span></dt><dd><p>Grow trees with <code class="docutils literal notranslate"><span class="pre">max_leaf_nodes</span></code> in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.</p>\n</dd>\n<dt>min_impurity_decrease<span class="classifier">float, default=0.0</span></dt><dd><p>A node will be split if this split induces a decrease of the impurity\ngreater than or equal to this value.</p>\n<p>The weighted impurity decrease equation is the following:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">N_t</span> <span class="o">/</span> <span class="n">N</span> <span class="o">*</span> <span class="p">(</span><span class="n">impurity</span> <span class="o">-</span> <span class="n">N_t_R</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">right_impurity</span>\n                    <span class="o">-</span> <span class="n">N_t_L</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">left_impurity</span><span class="p">)</span>\n</pre></div>\n</div>\n<p>where <code class="docutils literal notranslate"><span class="pre">N</span></code> is the total number of samples, <code class="docutils literal notranslate"><span class="pre">N_t</span></code> is the number of\nsamples at the current node, <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> is the number of samples in the\nleft child, and <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> is the number of samples in the right child.</p>\n<p><code class="docutils literal notranslate"><span class="pre">N</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> and <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> all refer to the weighted sum,\nif <code class="docutils literal notranslate"><span class="pre">sample_weight</span></code> is passed.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.19.</span></p>\n</div>\n</dd>\n<dt>bootstrap<span class="classifier">bool, default=True</span></dt><dd><p>Whether bootstrap samples are used when building trees. If False, the\nwhole dataset is used to build each tree.</p>\n</dd>\n<dt>oob_score<span class="classifier">bool or callable, default=False</span></dt><dd><p>Whether to use out-of-bag samples to estimate the generalization score.\nBy default, <code class="xref py py-func docutils literal notranslate"><span class="pre">r2_score()</span></code> is used.\nProvide a callable with signature <cite>metric(y_true, y_pred)</cite> to use a\ncustom metric. Only available if <cite>bootstrap=True</cite>.</p>\n</dd>\n<dt>n_jobs<span class="classifier">int, default=None</span></dt><dd><p>The number of jobs to run in parallel. <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code>, <code class="xref py py-meth docutils literal notranslate"><span class="pre">predict()</span></code>,\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">decision_path()</span></code> and <code class="xref py py-meth docutils literal notranslate"><span class="pre">apply()</span></code> are all parallelized over the\ntrees. <code class="docutils literal notranslate"><span class="pre">None</span></code> means 1 unless in a <code class="xref py py-obj docutils literal notranslate"><span class="pre">joblib.parallel_backend</span></code>\ncontext. <code class="docutils literal notranslate"><span class="pre">-1</span></code> means using all processors. See <span class="xref std std-term">Glossary</span> for more details.</p>\n</dd>\n<dt>random_state<span class="classifier">int, RandomState instance or None, default=None</span></dt><dd><p>Controls both the randomness of the bootstrapping of the samples used\nwhen building trees (if <code class="docutils literal notranslate"><span class="pre">bootstrap=True</span></code>) and the sampling of the\nfeatures to consider when looking for the best split at each node\n(if <code class="docutils literal notranslate"><span class="pre">max_features</span> <span class="pre">&lt;</span> <span class="pre">n_features</span></code>).\nSee <span class="xref std std-term">Glossary</span> for details.</p>\n</dd>\n<dt>verbose<span class="classifier">int, default=0</span></dt><dd><p>Controls the verbosity when fitting and predicting.</p>\n</dd>\n<dt>warm_start<span class="classifier">bool, default=False</span></dt><dd><p>When set to <code class="docutils literal notranslate"><span class="pre">True</span></code>, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest. See <span class="xref std std-term">Glossary</span> and\n<span class="xref std std-ref">gradient_boosting_warm_start</span> for details.</p>\n</dd>\n<dt>ccp_alpha<span class="classifier">non-negative float, default=0.0</span></dt><dd><p>Complexity parameter used for Minimal Cost-Complexity Pruning. The\nsubtree with the largest cost complexity that is smaller than\n<code class="docutils literal notranslate"><span class="pre">ccp_alpha</span></code> will be chosen. By default, no pruning is performed. See\n<span class="xref std std-ref">minimal_cost_complexity_pruning</span> for details.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n<dt>max_samples<span class="classifier">int or float, default=None</span></dt><dd><p>If bootstrap is True, the number of samples to draw from X\nto train each base estimator.</p>\n<ul class="simple">\n<li><p>If None (default), then draw <cite>X.shape[0]</cite> samples.</p></li>\n<li><p>If int, then draw <cite>max_samples</cite> samples.</p></li>\n<li><p>If float, then draw <cite>max(round(n_samples * max_samples), 1)</cite> samples. Thus,\n<cite>max_samples</cite> should be in the interval <cite>(0.0, 1.0]</cite>.</p></li>\n</ul>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id3"><span class="problematic" id="id4">estimator_</span></a><span class="classifier"><code class="xref py py-class docutils literal notranslate"><span class="pre">DecisionTreeRegressor</span></code></span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.2: </span><cite>base_estimator_</cite> was renamed to <cite>estimator_</cite>.</p>\n</div>\n</dd>\n<dt><a href="#id5"><span class="problematic" id="id6">base_estimator_</span></a><span class="classifier">DecisionTreeRegressor</span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="deprecated">\n<p><span class="versionmodified deprecated">Deprecated since version 1.2: </span><cite>base_estimator_</cite> is deprecated and will be removed in 1.4.\nUse <cite>estimator_</cite> instead.</p>\n</div>\n</dd>\n<dt><a href="#id7"><span class="problematic" id="id8">estimators_</span></a><span class="classifier">list of DecisionTreeRegressor</span></dt><dd><p>The collection of fitted sub-estimators.</p>\n</dd>\n<dt><a href="#id9"><span class="problematic" id="id10">feature_importances_</span></a><span class="classifier">ndarray of shape (n_features,)</span></dt><dd><p>The impurity-based feature importances.\nThe higher, the more important the feature.\nThe importance of a feature is computed as the (normalized)\ntotal reduction of the criterion brought by that feature.  It is also\nknown as the Gini importance.</p>\n<p>Warning: impurity-based feature importances can be misleading for\nhigh cardinality features (many unique values). See\n<code class="xref py py-func docutils literal notranslate"><span class="pre">sklearn.inspection.permutation_importance()</span></code> as an alternative.</p>\n</dd>\n<dt><a href="#id11"><span class="problematic" id="id12">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.24.</span></p>\n</div>\n</dd>\n<dt><a href="#id13"><span class="problematic" id="id14">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id15"><span class="problematic" id="id16">n_outputs_</span></a><span class="classifier">int</span></dt><dd><p>The number of outputs when <code class="docutils literal notranslate"><span class="pre">fit</span></code> is performed.</p>\n</dd>\n<dt><a href="#id17"><span class="problematic" id="id18">oob_score_</span></a><span class="classifier">float</span></dt><dd><p>Score of the training dataset obtained using an out-of-bag estimate.\nThis attribute exists only when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n<dt><a href="#id19"><span class="problematic" id="id20">oob_prediction_</span></a><span class="classifier">ndarray of shape (n_samples,) or (n_samples, n_outputs)</span></dt><dd><p>Prediction computed with out-of-bag estimate on the training set.\nThis attribute exists only when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n</dl>\n<p>sklearn.tree.DecisionTreeRegressor : A decision tree regressor.\nsklearn.ensemble.ExtraTreesRegressor : Ensemble of extremely randomized</p>\n<blockquote>\n<div><p>tree regressors.</p>\n</div></blockquote>\n<dl class="simple">\n<dt>sklearn.ensemble.HistGradientBoostingRegressor<span class="classifier">A Histogram-based Gradient</span></dt><dd><p>Boosting Regression Tree, very fast for big datasets (n_samples &gt;=\n10_000).</p>\n</dd>\n</dl>\n<p>The default values for the parameters controlling the size of the trees\n(e.g. <code class="docutils literal notranslate"><span class="pre">max_depth</span></code>, <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code>, etc.) lead to fully grown and\nunpruned trees which can potentially be very large on some data sets. To\nreduce memory consumption, the complexity and size of the trees should be\ncontrolled by setting those parameter values.</p>\n<p>The features are always randomly permuted at each split. Therefore,\nthe best found split may vary, even with the same training data,\n<code class="docutils literal notranslate"><span class="pre">max_features=n_features</span></code> and <code class="docutils literal notranslate"><span class="pre">bootstrap=False</span></code>, if the improvement\nof the criterion is identical for several splits enumerated during the\nsearch of the best split. To obtain a deterministic behaviour during\nfitting, <code class="docutils literal notranslate"><span class="pre">random_state</span></code> has to be fixed.</p>\n<p>The default value <code class="docutils literal notranslate"><span class="pre">max_features=1.0</span></code> uses <code class="docutils literal notranslate"><span class="pre">n_features</span></code>\nrather than <code class="docutils literal notranslate"><span class="pre">n_features</span> <span class="pre">/</span> <span class="pre">3</span></code>. The latter was originally suggested in\n[1], whereas the former was more recently justified empirically in [2].</p>\n<aside class="footnote brackets" id="id1" role="note">\n<span class="label"><span class="fn-bracket">[</span>1<span class="fn-bracket">]</span></span>\n<ol class="upperalpha simple" start="12">\n<li><p>Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.</p></li>\n</ol>\n</aside>\n<aside class="footnote brackets" id="id2" role="note">\n<span class="label"><span class="fn-bracket">[</span>2<span class="fn-bracket">]</span></span>\n<p>P. Geurts, D. Ernst., and L. Wehenkel, “Extremely randomized\ntrees”, Machine Learning, 63(1), 3-42, 2006.</p>\n</aside>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">RandomForestRegressor</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="kn">import</span> <span class="n">make_regression</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">make_regression</span><span class="p">(</span><span class="n">n_features</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">n_informative</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>\n<span class="gp">... </span>                       <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">regr</span> <span class="o">=</span> <span class="n">RandomForestRegressor</span><span class="p">(</span><span class="n">max_depth</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">regr</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>\n<span class="go">RandomForestRegressor(...)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">regr</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]]))</span>\n<span class="go">[-8.32987858]</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      n_estimators: 50,
      criterion: 'squared_error',
    },
  },
  {
    classPath: 'sklearn.ensemble.ExtraTreesRegressor',
    component: {
      constructorArgsSummary: {
        n_estimators: "<class 'int'>?",
        criterion: "<class 'str'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'scikit_reg',
    argsOptions: {
      max_features: [
        {
          key: 'sqrt',
          label: 'sqrt(n_features)',
          latex: '...',
        },
        {
          key: 'log2',
          label: 'log2(n_features)',
          latex: '...',
        },
      ],
      criterion: [
        {
          key: 'squared_error',
          label: 'Squared Error',
          latex: null,
        },
        {
          key: 'absolute_error',
          label: 'Absolute Error',
          latex: null,
        },
        {
          key: 'Friedman MSE',
          label: 'friedman_mse',
          latex: null,
        },
        {
          key: 'Poisson',
          label: 'poisson',
          latex: null,
        },
      ],
    },
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>An extra-trees regressor.</p>\n<p>This class implements a meta estimator that fits a number of\nrandomized decision trees (a.k.a. extra-trees) on various sub-samples\nof the dataset and uses averaging to improve the predictive accuracy\nand control over-fitting.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<dl>\n<dt>n_estimators<span class="classifier">int, default=100</span></dt><dd><p>The number of trees in the forest.</p>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.22: </span>The default value of <code class="docutils literal notranslate"><span class="pre">n_estimators</span></code> changed from 10 to 100\nin 0.22.</p>\n</div>\n</dd>\n<dt>criterion<span class="classifier">{“squared_error”, “absolute_error”, “friedman_mse”, “poisson”},             default=”squared_error”</span></dt><dd><p>The function to measure the quality of a split. Supported criteria\nare “squared_error” for the mean squared error, which is equal to\nvariance reduction as feature selection criterion and minimizes the L2\nloss using the mean of each terminal node, “friedman_mse”, which uses\nmean squared error with Friedman’s improvement score for potential\nsplits, “absolute_error” for the mean absolute error, which minimizes\nthe L1 loss using the median of each terminal node, and “poisson” which\nuses reduction in Poisson deviance to find splits.\nTraining using “absolute_error” is significantly slower\nthan when using “squared_error”.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.18: </span>Mean Absolute Error (MAE) criterion.</p>\n</div>\n</dd>\n<dt>max_depth<span class="classifier">int, default=None</span></dt><dd><p>The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.</p>\n</dd>\n<dt>min_samples_split<span class="classifier">int or float, default=2</span></dt><dd><p>The minimum number of samples required to split an internal node:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_split</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_split</cite> is a fraction and\n<cite>ceil(min_samples_split * n_samples)</cite> are the minimum\nnumber of samples for each split.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_samples_leaf<span class="classifier">int or float, default=1</span></dt><dd><p>The minimum number of samples required to be at a leaf node.\nA split point at any depth will only be considered if it leaves at\nleast <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code> training samples in each of the left and\nright branches.  This may have the effect of smoothing the model,\nespecially in regression.</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_leaf</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_leaf</cite> is a fraction and\n<cite>ceil(min_samples_leaf * n_samples)</cite> are the minimum\nnumber of samples for each node.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_weight_fraction_leaf<span class="classifier">float, default=0.0</span></dt><dd><p>The minimum weighted fraction of the sum total of weights (of all\nthe input samples) required to be at a leaf node. Samples have\nequal weight when sample_weight is not provided.</p>\n</dd>\n<dt>max_features<span class="classifier">{“sqrt”, “log2”, None}, int or float, default=1.0</span></dt><dd><p>The number of features to consider when looking for the best split:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>max_features</cite> features at each split.</p></li>\n<li><p>If float, then <cite>max_features</cite> is a fraction and\n<cite>max(1, int(max_features * n_features_in_))</cite> features are considered at each\nsplit.</p></li>\n<li><p>If “sqrt”, then <cite>max_features=sqrt(n_features)</cite>.</p></li>\n<li><p>If “log2”, then <cite>max_features=log2(n_features)</cite>.</p></li>\n<li><p>If None or 1.0, then <cite>max_features=n_features</cite>.</p></li>\n</ul>\n<div class="admonition note">\n<p class="admonition-title">Note</p>\n<p>The default of 1.0 is equivalent to bagged trees and more\nrandomness can be achieved by setting smaller values, e.g. 0.3.</p>\n</div>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 1.1: </span>The default of <cite>max_features</cite> changed from <cite>“auto”</cite> to 1.0.</p>\n</div>\n<p>Note: the search for a split does not stop until at least one\nvalid partition of the node samples is found, even if it requires to\neffectively inspect more than <code class="docutils literal notranslate"><span class="pre">max_features</span></code> features.</p>\n</dd>\n<dt>max_leaf_nodes<span class="classifier">int, default=None</span></dt><dd><p>Grow trees with <code class="docutils literal notranslate"><span class="pre">max_leaf_nodes</span></code> in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.</p>\n</dd>\n<dt>min_impurity_decrease<span class="classifier">float, default=0.0</span></dt><dd><p>A node will be split if this split induces a decrease of the impurity\ngreater than or equal to this value.</p>\n<p>The weighted impurity decrease equation is the following:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">N_t</span> <span class="o">/</span> <span class="n">N</span> <span class="o">*</span> <span class="p">(</span><span class="n">impurity</span> <span class="o">-</span> <span class="n">N_t_R</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">right_impurity</span>\n                    <span class="o">-</span> <span class="n">N_t_L</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">left_impurity</span><span class="p">)</span>\n</pre></div>\n</div>\n<p>where <code class="docutils literal notranslate"><span class="pre">N</span></code> is the total number of samples, <code class="docutils literal notranslate"><span class="pre">N_t</span></code> is the number of\nsamples at the current node, <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> is the number of samples in the\nleft child, and <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> is the number of samples in the right child.</p>\n<p><code class="docutils literal notranslate"><span class="pre">N</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> and <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> all refer to the weighted sum,\nif <code class="docutils literal notranslate"><span class="pre">sample_weight</span></code> is passed.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.19.</span></p>\n</div>\n</dd>\n<dt>bootstrap<span class="classifier">bool, default=False</span></dt><dd><p>Whether bootstrap samples are used when building trees. If False, the\nwhole dataset is used to build each tree.</p>\n</dd>\n<dt>oob_score<span class="classifier">bool or callable, default=False</span></dt><dd><p>Whether to use out-of-bag samples to estimate the generalization score.\nBy default, <code class="xref py py-func docutils literal notranslate"><span class="pre">r2_score()</span></code> is used.\nProvide a callable with signature <cite>metric(y_true, y_pred)</cite> to use a\ncustom metric. Only available if <cite>bootstrap=True</cite>.</p>\n</dd>\n<dt>n_jobs<span class="classifier">int, default=None</span></dt><dd><p>The number of jobs to run in parallel. <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code>, <code class="xref py py-meth docutils literal notranslate"><span class="pre">predict()</span></code>,\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">decision_path()</span></code> and <code class="xref py py-meth docutils literal notranslate"><span class="pre">apply()</span></code> are all parallelized over the\ntrees. <code class="docutils literal notranslate"><span class="pre">None</span></code> means 1 unless in a <code class="xref py py-obj docutils literal notranslate"><span class="pre">joblib.parallel_backend</span></code>\ncontext. <code class="docutils literal notranslate"><span class="pre">-1</span></code> means using all processors. See <span class="xref std std-term">Glossary</span> for more details.</p>\n</dd>\n<dt>random_state<span class="classifier">int, RandomState instance or None, default=None</span></dt><dd><p>Controls 3 sources of randomness:</p>\n<ul class="simple">\n<li><p>the bootstrapping of the samples used when building trees\n(if <code class="docutils literal notranslate"><span class="pre">bootstrap=True</span></code>)</p></li>\n<li><p>the sampling of the features to consider when looking for the best\nsplit at each node (if <code class="docutils literal notranslate"><span class="pre">max_features</span> <span class="pre">&lt;</span> <span class="pre">n_features</span></code>)</p></li>\n<li><p>the draw of the splits for each of the <cite>max_features</cite></p></li>\n</ul>\n<p>See <span class="xref std std-term">Glossary</span> for details.</p>\n</dd>\n<dt>verbose<span class="classifier">int, default=0</span></dt><dd><p>Controls the verbosity when fitting and predicting.</p>\n</dd>\n<dt>warm_start<span class="classifier">bool, default=False</span></dt><dd><p>When set to <code class="docutils literal notranslate"><span class="pre">True</span></code>, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest. See <span class="xref std std-term">Glossary</span> and\n<span class="xref std std-ref">gradient_boosting_warm_start</span> for details.</p>\n</dd>\n<dt>ccp_alpha<span class="classifier">non-negative float, default=0.0</span></dt><dd><p>Complexity parameter used for Minimal Cost-Complexity Pruning. The\nsubtree with the largest cost complexity that is smaller than\n<code class="docutils literal notranslate"><span class="pre">ccp_alpha</span></code> will be chosen. By default, no pruning is performed. See\n<span class="xref std std-ref">minimal_cost_complexity_pruning</span> for details.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n<dt>max_samples<span class="classifier">int or float, default=None</span></dt><dd><p>If bootstrap is True, the number of samples to draw from X\nto train each base estimator.</p>\n<ul class="simple">\n<li><p>If None (default), then draw <cite>X.shape[0]</cite> samples.</p></li>\n<li><p>If int, then draw <cite>max_samples</cite> samples.</p></li>\n<li><p>If float, then draw <cite>max_samples * X.shape[0]</cite> samples. Thus,\n<cite>max_samples</cite> should be in the interval <cite>(0.0, 1.0]</cite>.</p></li>\n</ul>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id2"><span class="problematic" id="id3">estimator_</span></a><span class="classifier"><code class="xref py py-class docutils literal notranslate"><span class="pre">ExtraTreeRegressor</span></code></span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.2: </span><cite>base_estimator_</cite> was renamed to <cite>estimator_</cite>.</p>\n</div>\n</dd>\n<dt><a href="#id4"><span class="problematic" id="id5">base_estimator_</span></a><span class="classifier">ExtraTreeRegressor</span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="deprecated">\n<p><span class="versionmodified deprecated">Deprecated since version 1.2: </span><cite>base_estimator_</cite> is deprecated and will be removed in 1.4.\nUse <cite>estimator_</cite> instead.</p>\n</div>\n</dd>\n<dt><a href="#id6"><span class="problematic" id="id7">estimators_</span></a><span class="classifier">list of DecisionTreeRegressor</span></dt><dd><p>The collection of fitted sub-estimators.</p>\n</dd>\n<dt><a href="#id8"><span class="problematic" id="id9">feature_importances_</span></a><span class="classifier">ndarray of shape (n_features,)</span></dt><dd><p>The impurity-based feature importances.\nThe higher, the more important the feature.\nThe importance of a feature is computed as the (normalized)\ntotal reduction of the criterion brought by that feature.  It is also\nknown as the Gini importance.</p>\n<p>Warning: impurity-based feature importances can be misleading for\nhigh cardinality features (many unique values). See\n<code class="xref py py-func docutils literal notranslate"><span class="pre">sklearn.inspection.permutation_importance()</span></code> as an alternative.</p>\n</dd>\n<dt><a href="#id10"><span class="problematic" id="id11">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.24.</span></p>\n</div>\n</dd>\n<dt><a href="#id12"><span class="problematic" id="id13">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id14"><span class="problematic" id="id15">n_outputs_</span></a><span class="classifier">int</span></dt><dd><p>The number of outputs.</p>\n</dd>\n<dt><a href="#id16"><span class="problematic" id="id17">oob_score_</span></a><span class="classifier">float</span></dt><dd><p>Score of the training dataset obtained using an out-of-bag estimate.\nThis attribute exists only when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n<dt><a href="#id18"><span class="problematic" id="id19">oob_prediction_</span></a><span class="classifier">ndarray of shape (n_samples,) or (n_samples, n_outputs)</span></dt><dd><p>Prediction computed with out-of-bag estimate on the training set.\nThis attribute exists only when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n</dl>\n<p>ExtraTreesClassifier : An extra-trees classifier with random splits.\nRandomForestClassifier : A random forest classifier with optimal splits.\nRandomForestRegressor : Ensemble regressor using trees with optimal splits.</p>\n<p>The default values for the parameters controlling the size of the trees\n(e.g. <code class="docutils literal notranslate"><span class="pre">max_depth</span></code>, <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code>, etc.) lead to fully grown and\nunpruned trees which can potentially be very large on some data sets. To\nreduce memory consumption, the complexity and size of the trees should be\ncontrolled by setting those parameter values.</p>\n<aside class="footnote brackets" id="id1" role="note">\n<span class="label"><span class="fn-bracket">[</span>1<span class="fn-bracket">]</span></span>\n<p>P. Geurts, D. Ernst., and L. Wehenkel, “Extremely randomized trees”,\nMachine Learning, 63(1), 3-42, 2006.</p>\n</aside>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="kn">import</span> <span class="n">load_diabetes</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">ExtraTreesRegressor</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">load_diabetes</span><span class="p">(</span><span class="n">return_X_y</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span>\n<span class="gp">... </span>    <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">reg</span> <span class="o">=</span> <span class="n">ExtraTreesRegressor</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span>\n<span class="gp">... </span>   <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">reg</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>\n<span class="go">0.2727...</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      n_estimators: 100,
      criterion: 'squared_error',
    },
  },
  {
    classPath: 'sklearn.ensemble.ExtraTreesClassifier',
    component: {
      constructorArgsSummary: {
        n_estimators: "<class 'int'>?",
        criterion: "<class 'str'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'scikit_class',
    argsOptions: {
      max_features: [
        {
          key: 'sqrt',
          label: 'sqrt(n_features)',
          latex: '...',
        },
        {
          key: 'log2',
          label: 'log2(n_features)',
          latex: '...',
        },
      ],
      criterion: [
        {
          key: 'gini',
          label: 'Gini impurity',
          latex: '...',
        },
        {
          key: 'entropy',
          label: 'Entropy',
          latex: '...',
        },
        {
          key: 'log_loss',
          label: 'Log loss',
          latex: '...',
        },
      ],
      class_weight: [
        {
          key: 'balanced',
          label: 'Balanced',
          latex: null,
        },
        {
          key: 'balanced_subsample',
          label: 'Balanced Subsample',
          latex: null,
        },
      ],
    },
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>An extra-trees classifier.</p>\n<p>This class implements a meta estimator that fits a number of\nrandomized decision trees (a.k.a. extra-trees) on various sub-samples\nof the dataset and uses averaging to improve the predictive accuracy\nand control over-fitting.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<dl>\n<dt>n_estimators<span class="classifier">int, default=100</span></dt><dd><p>The number of trees in the forest.</p>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.22: </span>The default value of <code class="docutils literal notranslate"><span class="pre">n_estimators</span></code> changed from 10 to 100\nin 0.22.</p>\n</div>\n</dd>\n<dt>criterion<span class="classifier">{“gini”, “entropy”, “log_loss”}, default=”gini”</span></dt><dd><p>The function to measure the quality of a split. Supported criteria are\n“gini” for the Gini impurity and “log_loss” and “entropy” both for the\nShannon information gain, see <span class="xref std std-ref">tree_mathematical_formulation</span>.\nNote: This parameter is tree-specific.</p>\n</dd>\n<dt>max_depth<span class="classifier">int, default=None</span></dt><dd><p>The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.</p>\n</dd>\n<dt>min_samples_split<span class="classifier">int or float, default=2</span></dt><dd><p>The minimum number of samples required to split an internal node:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_split</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_split</cite> is a fraction and\n<cite>ceil(min_samples_split * n_samples)</cite> are the minimum\nnumber of samples for each split.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_samples_leaf<span class="classifier">int or float, default=1</span></dt><dd><p>The minimum number of samples required to be at a leaf node.\nA split point at any depth will only be considered if it leaves at\nleast <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code> training samples in each of the left and\nright branches.  This may have the effect of smoothing the model,\nespecially in regression.</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_leaf</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_leaf</cite> is a fraction and\n<cite>ceil(min_samples_leaf * n_samples)</cite> are the minimum\nnumber of samples for each node.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_weight_fraction_leaf<span class="classifier">float, default=0.0</span></dt><dd><p>The minimum weighted fraction of the sum total of weights (of all\nthe input samples) required to be at a leaf node. Samples have\nequal weight when sample_weight is not provided.</p>\n</dd>\n<dt>max_features<span class="classifier">{“sqrt”, “log2”, None}, int or float, default=”sqrt”</span></dt><dd><p>The number of features to consider when looking for the best split:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>max_features</cite> features at each split.</p></li>\n<li><p>If float, then <cite>max_features</cite> is a fraction and\n<cite>max(1, int(max_features * n_features_in_))</cite> features are considered at each\nsplit.</p></li>\n<li><p>If “sqrt”, then <cite>max_features=sqrt(n_features)</cite>.</p></li>\n<li><p>If “log2”, then <cite>max_features=log2(n_features)</cite>.</p></li>\n<li><p>If None, then <cite>max_features=n_features</cite>.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 1.1: </span>The default of <cite>max_features</cite> changed from <cite>“auto”</cite> to <cite>“sqrt”</cite>.</p>\n</div>\n<p>Note: the search for a split does not stop until at least one\nvalid partition of the node samples is found, even if it requires to\neffectively inspect more than <code class="docutils literal notranslate"><span class="pre">max_features</span></code> features.</p>\n</dd>\n<dt>max_leaf_nodes<span class="classifier">int, default=None</span></dt><dd><p>Grow trees with <code class="docutils literal notranslate"><span class="pre">max_leaf_nodes</span></code> in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.</p>\n</dd>\n<dt>min_impurity_decrease<span class="classifier">float, default=0.0</span></dt><dd><p>A node will be split if this split induces a decrease of the impurity\ngreater than or equal to this value.</p>\n<p>The weighted impurity decrease equation is the following:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">N_t</span> <span class="o">/</span> <span class="n">N</span> <span class="o">*</span> <span class="p">(</span><span class="n">impurity</span> <span class="o">-</span> <span class="n">N_t_R</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">right_impurity</span>\n                    <span class="o">-</span> <span class="n">N_t_L</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">left_impurity</span><span class="p">)</span>\n</pre></div>\n</div>\n<p>where <code class="docutils literal notranslate"><span class="pre">N</span></code> is the total number of samples, <code class="docutils literal notranslate"><span class="pre">N_t</span></code> is the number of\nsamples at the current node, <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> is the number of samples in the\nleft child, and <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> is the number of samples in the right child.</p>\n<p><code class="docutils literal notranslate"><span class="pre">N</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> and <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> all refer to the weighted sum,\nif <code class="docutils literal notranslate"><span class="pre">sample_weight</span></code> is passed.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.19.</span></p>\n</div>\n</dd>\n<dt>bootstrap<span class="classifier">bool, default=False</span></dt><dd><p>Whether bootstrap samples are used when building trees. If False, the\nwhole dataset is used to build each tree.</p>\n</dd>\n<dt>oob_score<span class="classifier">bool or callable, default=False</span></dt><dd><p>Whether to use out-of-bag samples to estimate the generalization score.\nBy default, <code class="xref py py-func docutils literal notranslate"><span class="pre">accuracy_score()</span></code> is used.\nProvide a callable with signature <cite>metric(y_true, y_pred)</cite> to use a\ncustom metric. Only available if <cite>bootstrap=True</cite>.</p>\n</dd>\n<dt>n_jobs<span class="classifier">int, default=None</span></dt><dd><p>The number of jobs to run in parallel. <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code>, <code class="xref py py-meth docutils literal notranslate"><span class="pre">predict()</span></code>,\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">decision_path()</span></code> and <code class="xref py py-meth docutils literal notranslate"><span class="pre">apply()</span></code> are all parallelized over the\ntrees. <code class="docutils literal notranslate"><span class="pre">None</span></code> means 1 unless in a <code class="xref py py-obj docutils literal notranslate"><span class="pre">joblib.parallel_backend</span></code>\ncontext. <code class="docutils literal notranslate"><span class="pre">-1</span></code> means using all processors. See <span class="xref std std-term">Glossary</span> for more details.</p>\n</dd>\n<dt>random_state<span class="classifier">int, RandomState instance or None, default=None</span></dt><dd><p>Controls 3 sources of randomness:</p>\n<ul class="simple">\n<li><p>the bootstrapping of the samples used when building trees\n(if <code class="docutils literal notranslate"><span class="pre">bootstrap=True</span></code>)</p></li>\n<li><p>the sampling of the features to consider when looking for the best\nsplit at each node (if <code class="docutils literal notranslate"><span class="pre">max_features</span> <span class="pre">&lt;</span> <span class="pre">n_features</span></code>)</p></li>\n<li><p>the draw of the splits for each of the <cite>max_features</cite></p></li>\n</ul>\n<p>See <span class="xref std std-term">Glossary</span> for details.</p>\n</dd>\n<dt>verbose<span class="classifier">int, default=0</span></dt><dd><p>Controls the verbosity when fitting and predicting.</p>\n</dd>\n<dt>warm_start<span class="classifier">bool, default=False</span></dt><dd><p>When set to <code class="docutils literal notranslate"><span class="pre">True</span></code>, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest. See <span class="xref std std-term">Glossary</span> and\n<span class="xref std std-ref">gradient_boosting_warm_start</span> for details.</p>\n</dd>\n<dt>class_weight<span class="classifier">{“balanced”, “balanced_subsample”}, dict or list of dicts,             default=None</span></dt><dd><p>Weights associated with classes in the form <code class="docutils literal notranslate"><span class="pre">{class_label:</span> <span class="pre">weight}</span></code>.\nIf not given, all classes are supposed to have weight one. For\nmulti-output problems, a list of dicts can be provided in the same\norder as the columns of y.</p>\n<p>Note that for multioutput (including multilabel) weights should be\ndefined for each class of every column in its own dict. For example,\nfor four-class multilabel classification weights should be\n[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of\n[{1:1}, {2:5}, {3:1}, {4:1}].</p>\n<p>The “balanced” mode uses the values of y to automatically adjust\nweights inversely proportional to class frequencies in the input data\nas <code class="docutils literal notranslate"><span class="pre">n_samples</span> <span class="pre">/</span> <span class="pre">(n_classes</span> <span class="pre">*</span> <span class="pre">np.bincount(y))</span></code></p>\n<p>The “balanced_subsample” mode is the same as “balanced” except that\nweights are computed based on the bootstrap sample for every tree\ngrown.</p>\n<p>For multi-output, the weights of each column of y will be multiplied.</p>\n<p>Note that these weights will be multiplied with sample_weight (passed\nthrough the fit method) if sample_weight is specified.</p>\n</dd>\n<dt>ccp_alpha<span class="classifier">non-negative float, default=0.0</span></dt><dd><p>Complexity parameter used for Minimal Cost-Complexity Pruning. The\nsubtree with the largest cost complexity that is smaller than\n<code class="docutils literal notranslate"><span class="pre">ccp_alpha</span></code> will be chosen. By default, no pruning is performed. See\n<span class="xref std std-ref">minimal_cost_complexity_pruning</span> for details.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n<dt>max_samples<span class="classifier">int or float, default=None</span></dt><dd><p>If bootstrap is True, the number of samples to draw from X\nto train each base estimator.</p>\n<ul class="simple">\n<li><p>If None (default), then draw <cite>X.shape[0]</cite> samples.</p></li>\n<li><p>If int, then draw <cite>max_samples</cite> samples.</p></li>\n<li><p>If float, then draw <cite>max_samples * X.shape[0]</cite> samples. Thus,\n<cite>max_samples</cite> should be in the interval <cite>(0.0, 1.0]</cite>.</p></li>\n</ul>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id2"><span class="problematic" id="id3">estimator_</span></a><span class="classifier"><code class="xref py py-class docutils literal notranslate"><span class="pre">ExtraTreesClassifier</span></code></span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.2: </span><cite>base_estimator_</cite> was renamed to <cite>estimator_</cite>.</p>\n</div>\n</dd>\n<dt><a href="#id4"><span class="problematic" id="id5">base_estimator_</span></a><span class="classifier">ExtraTreesClassifier</span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="deprecated">\n<p><span class="versionmodified deprecated">Deprecated since version 1.2: </span><cite>base_estimator_</cite> is deprecated and will be removed in 1.4.\nUse <cite>estimator_</cite> instead.</p>\n</div>\n</dd>\n<dt><a href="#id6"><span class="problematic" id="id7">estimators_</span></a><span class="classifier">list of DecisionTreeClassifier</span></dt><dd><p>The collection of fitted sub-estimators.</p>\n</dd>\n<dt><a href="#id8"><span class="problematic" id="id9">classes_</span></a><span class="classifier">ndarray of shape (n_classes,) or a list of such arrays</span></dt><dd><p>The classes labels (single output problem), or a list of arrays of\nclass labels (multi-output problem).</p>\n</dd>\n<dt><a href="#id10"><span class="problematic" id="id11">n_classes_</span></a><span class="classifier">int or list</span></dt><dd><p>The number of classes (single output problem), or a list containing the\nnumber of classes for each output (multi-output problem).</p>\n</dd>\n<dt><a href="#id12"><span class="problematic" id="id13">feature_importances_</span></a><span class="classifier">ndarray of shape (n_features,)</span></dt><dd><p>The impurity-based feature importances.\nThe higher, the more important the feature.\nThe importance of a feature is computed as the (normalized)\ntotal reduction of the criterion brought by that feature.  It is also\nknown as the Gini importance.</p>\n<p>Warning: impurity-based feature importances can be misleading for\nhigh cardinality features (many unique values). See\n<code class="xref py py-func docutils literal notranslate"><span class="pre">sklearn.inspection.permutation_importance()</span></code> as an alternative.</p>\n</dd>\n<dt><a href="#id14"><span class="problematic" id="id15">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.24.</span></p>\n</div>\n</dd>\n<dt><a href="#id16"><span class="problematic" id="id17">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id18"><span class="problematic" id="id19">n_outputs_</span></a><span class="classifier">int</span></dt><dd><p>The number of outputs when <code class="docutils literal notranslate"><span class="pre">fit</span></code> is performed.</p>\n</dd>\n<dt><a href="#id20"><span class="problematic" id="id21">oob_score_</span></a><span class="classifier">float</span></dt><dd><p>Score of the training dataset obtained using an out-of-bag estimate.\nThis attribute exists only when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n<dt><a href="#id22"><span class="problematic" id="id23">oob_decision_function_</span></a><span class="classifier">ndarray of shape (n_samples, n_classes) or             (n_samples, n_classes, n_outputs)</span></dt><dd><p>Decision function computed with out-of-bag estimate on the training\nset. If n_estimators is small it might be possible that a data point\nwas never left out during the bootstrap. In this case,\n<cite>oob_decision_function_</cite> might contain NaN. This attribute exists\nonly when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n</dl>\n<p>ExtraTreesRegressor : An extra-trees regressor with random splits.\nRandomForestClassifier : A random forest classifier with optimal splits.\nRandomForestRegressor : Ensemble regressor using trees with optimal splits.</p>\n<p>The default values for the parameters controlling the size of the trees\n(e.g. <code class="docutils literal notranslate"><span class="pre">max_depth</span></code>, <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code>, etc.) lead to fully grown and\nunpruned trees which can potentially be very large on some data sets. To\nreduce memory consumption, the complexity and size of the trees should be\ncontrolled by setting those parameter values.</p>\n<aside class="footnote brackets" id="id1" role="note">\n<span class="label"><span class="fn-bracket">[</span>1<span class="fn-bracket">]</span></span>\n<p>P. Geurts, D. Ernst., and L. Wehenkel, “Extremely randomized\ntrees”, Machine Learning, 63(1), 3-42, 2006.</p>\n</aside>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">ExtraTreesClassifier</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="kn">import</span> <span class="n">make_classification</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">make_classification</span><span class="p">(</span><span class="n">n_features</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">clf</span> <span class="o">=</span> <span class="n">ExtraTreesClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>\n<span class="go">ExtraTreesClassifier(random_state=0)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]])</span>\n<span class="go">array([1])</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      n_estimators: 100,
      criterion: 'gini',
    },
  },
  {
    classPath: 'sklearn.neighbors.KNeighborsClassifier',
    component: {
      constructorArgsSummary: {
        n_neighbors: "<class 'int'>?",
        weights: "<class 'str'>?",
        algorithm: "<class 'str'>?",
        leaf_size: "<class 'int'>?",
        p: "<class 'int'>?",
        metric: "<class 'str'>?",
        metric_params: "<class 'NoneType'>?",
        n_jobs: "<class 'NoneType'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'scikit_class',
    argsOptions: {
      weights: [
        {
          key: 'uniform',
          label: null,
          latex: null,
        },
        {
          key: 'distance',
          label: null,
          latex: null,
        },
      ],
      algorithm: [
        {
          key: 'auto',
          label: null,
          latex: null,
        },
        {
          key: 'ball_tree',
          label: null,
          latex: null,
        },
        {
          key: 'kd_tree',
          label: null,
          latex: null,
        },
        {
          key: 'brute',
          label: null,
          latex: null,
        },
      ],
      metric: [
        {
          key: 'l1',
          label: 'cityblock (l1)',
          latex: '...',
        },
        {
          key: 'l2',
          label: 'euclidean (l2)',
          latex: '...',
        },
        {
          key: 'haversine',
          label: 'haversine',
          latex: '...',
        },
        {
          key: 'cosine',
          label: 'cosine',
          latex: '...',
        },
      ],
    },
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <p>Classifier implementing the k-nearest neighbors vote.</p>\n<blockquote>\n<div><p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<dl>\n<dt>n_neighbors<span class="classifier">int, default=5</span></dt><dd><p>Number of neighbors to use by default for <code class="xref py py-meth docutils literal notranslate"><span class="pre">kneighbors()</span></code> queries.</p>\n</dd>\n<dt>weights<span class="classifier">{‘uniform’, ‘distance’}, callable or None, default=’uniform’</span></dt><dd><p>Weight function used in prediction.  Possible values:</p>\n<ul class="simple">\n<li><p>‘uniform’ : uniform weights.  All points in each neighborhood\nare weighted equally.</p></li>\n<li><p>‘distance’ : weight points by the inverse of their distance.\nin this case, closer neighbors of a query point will have a\ngreater influence than neighbors which are further away.</p></li>\n<li><p>[callable] : a user-defined function which accepts an\narray of distances, and returns an array of the same shape\ncontaining the weights.</p></li>\n</ul>\n</dd>\n<dt>algorithm<span class="classifier">{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’</span></dt><dd><p>Algorithm used to compute the nearest neighbors:</p>\n<ul class="simple">\n<li><p>‘ball_tree’ will use <code class="xref py py-class docutils literal notranslate"><span class="pre">BallTree</span></code></p></li>\n<li><p>‘kd_tree’ will use <code class="xref py py-class docutils literal notranslate"><span class="pre">KDTree</span></code></p></li>\n<li><p>‘brute’ will use a brute-force search.</p></li>\n<li><p>‘auto’ will attempt to decide the most appropriate algorithm\nbased on the values passed to <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code> method.</p></li>\n</ul>\n<p>Note: fitting on sparse input will override the setting of\nthis parameter, using brute force.</p>\n</dd>\n<dt>leaf_size<span class="classifier">int, default=30</span></dt><dd><p>Leaf size passed to BallTree or KDTree.  This can affect the\nspeed of the construction and query, as well as the memory\nrequired to store the tree.  The optimal value depends on the\nnature of the problem.</p>\n</dd>\n<dt>p<span class="classifier">int, default=2</span></dt><dd><p>Power parameter for the Minkowski metric. When p = 1, this is\nequivalent to using manhattan_distance (l1), and euclidean_distance\n(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.</p>\n</dd>\n<dt>metric<span class="classifier">str or callable, default=’minkowski’</span></dt><dd><p>Metric to use for distance computation. Default is “minkowski”, which\nresults in the standard Euclidean distance when p = 2. See the\ndocumentation of <a class="reference external" href="https://docs.scipy.org/doc/scipy/reference/spatial.distance.html">scipy.spatial.distance</a> and\nthe metrics listed in\n<code class="xref py py-class docutils literal notranslate"><span class="pre">distance_metrics</span></code> for valid metric\nvalues.</p>\n<p>If metric is “precomputed”, X is assumed to be a distance matrix and\nmust be square during fit. X may be a <span class="xref std std-term">sparse graph</span>, in which\ncase only “nonzero” elements may be considered neighbors.</p>\n<p>If metric is a callable function, it takes two arrays representing 1D\nvectors as inputs and must return one value indicating the distance\nbetween those vectors. This works for Scipy’s metrics, but is less\nefficient than passing the metric name as a string.</p>\n</dd>\n<dt>metric_params<span class="classifier">dict, default=None</span></dt><dd><p>Additional keyword arguments for the metric function.</p>\n</dd>\n<dt>n_jobs<span class="classifier">int, default=None</span></dt><dd><p>The number of parallel jobs to run for neighbors search.\n<code class="docutils literal notranslate"><span class="pre">None</span></code> means 1 unless in a <code class="xref py py-obj docutils literal notranslate"><span class="pre">joblib.parallel_backend</span></code> context.\n<code class="docutils literal notranslate"><span class="pre">-1</span></code> means using all processors. See <span class="xref std std-term">Glossary</span>\nfor more details.\nDoesn’t affect <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code> method.</p>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id1"><span class="problematic" id="id2">classes_</span></a><span class="classifier">array of shape (n_classes,)</span></dt><dd><p>Class labels known to the classifier</p>\n</dd>\n<dt><a href="#id3"><span class="problematic" id="id4">effective_metric_</span></a><span class="classifier">str or callble</span></dt><dd><p>The distance metric used. It will be same as the <cite>metric</cite> parameter\nor a synonym of it, e.g. ‘euclidean’ if the <cite>metric</cite> parameter set to\n‘minkowski’ and <cite>p</cite> parameter set to 2.</p>\n</dd>\n<dt><a href="#id5"><span class="problematic" id="id6">effective_metric_params_</span></a><span class="classifier">dict</span></dt><dd><p>Additional keyword arguments for the metric function. For most metrics\nwill be same with <cite>metric_params</cite> parameter, but may also contain the\n<cite>p</cite> parameter value if the <cite>effective_metric_</cite> attribute is set to\n‘minkowski’.</p>\n</dd>\n<dt><a href="#id7"><span class="problematic" id="id8">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.24.</span></p>\n</div>\n</dd>\n<dt><a href="#id9"><span class="problematic" id="id10">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id11"><span class="problematic" id="id12">n_samples_fit_</span></a><span class="classifier">int</span></dt><dd><p>Number of samples in the fitted data.</p>\n</dd>\n<dt><a href="#id13"><span class="problematic" id="id14">outputs_2d_</span></a><span class="classifier">bool</span></dt><dd><p>False when <cite>y</cite>’s shape is (n_samples, ) or (n_samples, 1) during fit\notherwise True.</p>\n</dd>\n</dl>\n<p>RadiusNeighborsClassifier: Classifier based on neighbors within a fixed radius.\nKNeighborsRegressor: Regression based on k-nearest neighbors.\nRadiusNeighborsRegressor: Regression based on neighbors within a fixed radius.\nNearestNeighbors: Unsupervised learner for implementing neighbor searches.</p>\n<p>See <span class="xref std std-ref">Nearest Neighbors</span> in the online documentation\nfor a discussion of the choice of <code class="docutils literal notranslate"><span class="pre">algorithm</span></code> and <code class="docutils literal notranslate"><span class="pre">leaf_size</span></code>.</p>\n<div class="admonition warning">\n<p class="admonition-title">Warning</p>\n<p>Regarding the Nearest Neighbors algorithms, if it is found that two\nneighbors, neighbor <cite>k+1</cite> and <cite>k</cite>, have identical distances\nbut different labels, the results will depend on the ordering of the\ntraining data.</p>\n</div>\n<p><a class="reference external" href="https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm">https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm</a></p>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">X</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">2</span><span class="p">],</span> <span class="p">[</span><span class="mi">3</span><span class="p">]]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">y</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">neigh</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">neigh</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>\n<span class="go">KNeighborsClassifier(...)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">neigh</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mf">1.1</span><span class="p">]]))</span>\n<span class="go">[0]</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">neigh</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">([[</span><span class="mf">0.9</span><span class="p">]]))</span>\n<span class="go">[[0.666... 0.333...]]</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      n_neighbors: 5,
      weights: 'uniform',
      algorithm: 'auto',
      leaf_size: 30,
      p: 2,
      metric: 'minkowski',
      metric_params: null,
      n_jobs: null,
    },
  },
  {
    classPath: 'sklearn.ensemble.RandomForestClassifier',
    component: {
      constructorArgsSummary: {
        n_estimators: "<class 'int'>?",
        criterion: "<class 'str'>?",
      },
      forwardArgsSummary: {},
    },
    type: 'scikit_class',
    argsOptions: {
      max_features: [
        {
          key: 'sqrt',
          label: 'sqrt(n_features)',
          latex: '...',
        },
        {
          key: 'log2',
          label: 'log2(n_features)',
          latex: '...',
        },
      ],
      criterion: [
        {
          key: 'gini',
          label: 'Gini impurity',
          latex: '...',
        },
        {
          key: 'entropy',
          label: 'Entropy',
          latex: '...',
        },
        {
          key: 'log_loss',
          label: 'Log loss',
          latex: '...',
        },
      ],
    },
    docsLink: null,
    docs: '<div class="docstring">\n    \n  <blockquote>\n<div><p>A random forest classifier.</p>\n<p>A random forest is a meta estimator that fits a number of decision tree\nclassifiers on various sub-samples of the dataset and uses averaging to\nimprove the predictive accuracy and control over-fitting.\nThe sub-sample size is controlled with the <cite>max_samples</cite> parameter if\n<cite>bootstrap=True</cite> (default), otherwise the whole dataset is used to build\neach tree.</p>\n<p>For a comparison between tree-based ensemble models see the example\n<span class="xref std std-ref">sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py</span>.</p>\n<p>Read more in the <span class="xref std std-ref">User Guide</span>.</p>\n<dl>\n<dt>n_estimators<span class="classifier">int, default=100</span></dt><dd><p>The number of trees in the forest.</p>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.22: </span>The default value of <code class="docutils literal notranslate"><span class="pre">n_estimators</span></code> changed from 10 to 100\nin 0.22.</p>\n</div>\n</dd>\n<dt>criterion<span class="classifier">{“gini”, “entropy”, “log_loss”}, default=”gini”</span></dt><dd><p>The function to measure the quality of a split. Supported criteria are\n“gini” for the Gini impurity and “log_loss” and “entropy” both for the\nShannon information gain, see <span class="xref std std-ref">tree_mathematical_formulation</span>.\nNote: This parameter is tree-specific.</p>\n</dd>\n<dt>max_depth<span class="classifier">int, default=None</span></dt><dd><p>The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.</p>\n</dd>\n<dt>min_samples_split<span class="classifier">int or float, default=2</span></dt><dd><p>The minimum number of samples required to split an internal node:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_split</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_split</cite> is a fraction and\n<cite>ceil(min_samples_split * n_samples)</cite> are the minimum\nnumber of samples for each split.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_samples_leaf<span class="classifier">int or float, default=1</span></dt><dd><p>The minimum number of samples required to be at a leaf node.\nA split point at any depth will only be considered if it leaves at\nleast <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code> training samples in each of the left and\nright branches.  This may have the effect of smoothing the model,\nespecially in regression.</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>min_samples_leaf</cite> as the minimum number.</p></li>\n<li><p>If float, then <cite>min_samples_leaf</cite> is a fraction and\n<cite>ceil(min_samples_leaf * n_samples)</cite> are the minimum\nnumber of samples for each node.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 0.18: </span>Added float values for fractions.</p>\n</div>\n</dd>\n<dt>min_weight_fraction_leaf<span class="classifier">float, default=0.0</span></dt><dd><p>The minimum weighted fraction of the sum total of weights (of all\nthe input samples) required to be at a leaf node. Samples have\nequal weight when sample_weight is not provided.</p>\n</dd>\n<dt>max_features<span class="classifier">{“sqrt”, “log2”, None}, int or float, default=”sqrt”</span></dt><dd><p>The number of features to consider when looking for the best split:</p>\n<ul class="simple">\n<li><p>If int, then consider <cite>max_features</cite> features at each split.</p></li>\n<li><p>If float, then <cite>max_features</cite> is a fraction and\n<cite>max(1, int(max_features * n_features_in_))</cite> features are considered at each\nsplit.</p></li>\n<li><p>If “sqrt”, then <cite>max_features=sqrt(n_features)</cite>.</p></li>\n<li><p>If “log2”, then <cite>max_features=log2(n_features)</cite>.</p></li>\n<li><p>If None, then <cite>max_features=n_features</cite>.</p></li>\n</ul>\n<div class="versionchanged">\n<p><span class="versionmodified changed">Changed in version 1.1: </span>The default of <cite>max_features</cite> changed from <cite>“auto”</cite> to <cite>“sqrt”</cite>.</p>\n</div>\n<p>Note: the search for a split does not stop until at least one\nvalid partition of the node samples is found, even if it requires to\neffectively inspect more than <code class="docutils literal notranslate"><span class="pre">max_features</span></code> features.</p>\n</dd>\n<dt>max_leaf_nodes<span class="classifier">int, default=None</span></dt><dd><p>Grow trees with <code class="docutils literal notranslate"><span class="pre">max_leaf_nodes</span></code> in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.</p>\n</dd>\n<dt>min_impurity_decrease<span class="classifier">float, default=0.0</span></dt><dd><p>A node will be split if this split induces a decrease of the impurity\ngreater than or equal to this value.</p>\n<p>The weighted impurity decrease equation is the following:</p>\n<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">N_t</span> <span class="o">/</span> <span class="n">N</span> <span class="o">*</span> <span class="p">(</span><span class="n">impurity</span> <span class="o">-</span> <span class="n">N_t_R</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">right_impurity</span>\n                    <span class="o">-</span> <span class="n">N_t_L</span> <span class="o">/</span> <span class="n">N_t</span> <span class="o">*</span> <span class="n">left_impurity</span><span class="p">)</span>\n</pre></div>\n</div>\n<p>where <code class="docutils literal notranslate"><span class="pre">N</span></code> is the total number of samples, <code class="docutils literal notranslate"><span class="pre">N_t</span></code> is the number of\nsamples at the current node, <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> is the number of samples in the\nleft child, and <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> is the number of samples in the right child.</p>\n<p><code class="docutils literal notranslate"><span class="pre">N</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t</span></code>, <code class="docutils literal notranslate"><span class="pre">N_t_R</span></code> and <code class="docutils literal notranslate"><span class="pre">N_t_L</span></code> all refer to the weighted sum,\nif <code class="docutils literal notranslate"><span class="pre">sample_weight</span></code> is passed.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.19.</span></p>\n</div>\n</dd>\n<dt>bootstrap<span class="classifier">bool, default=True</span></dt><dd><p>Whether bootstrap samples are used when building trees. If False, the\nwhole dataset is used to build each tree.</p>\n</dd>\n<dt>oob_score<span class="classifier">bool or callable, default=False</span></dt><dd><p>Whether to use out-of-bag samples to estimate the generalization score.\nBy default, <code class="xref py py-func docutils literal notranslate"><span class="pre">accuracy_score()</span></code> is used.\nProvide a callable with signature <cite>metric(y_true, y_pred)</cite> to use a\ncustom metric. Only available if <cite>bootstrap=True</cite>.</p>\n</dd>\n<dt>n_jobs<span class="classifier">int, default=None</span></dt><dd><p>The number of jobs to run in parallel. <code class="xref py py-meth docutils literal notranslate"><span class="pre">fit()</span></code>, <code class="xref py py-meth docutils literal notranslate"><span class="pre">predict()</span></code>,\n<code class="xref py py-meth docutils literal notranslate"><span class="pre">decision_path()</span></code> and <code class="xref py py-meth docutils literal notranslate"><span class="pre">apply()</span></code> are all parallelized over the\ntrees. <code class="docutils literal notranslate"><span class="pre">None</span></code> means 1 unless in a <code class="xref py py-obj docutils literal notranslate"><span class="pre">joblib.parallel_backend</span></code>\ncontext. <code class="docutils literal notranslate"><span class="pre">-1</span></code> means using all processors. See <span class="xref std std-term">Glossary</span> for more details.</p>\n</dd>\n<dt>random_state<span class="classifier">int, RandomState instance or None, default=None</span></dt><dd><p>Controls both the randomness of the bootstrapping of the samples used\nwhen building trees (if <code class="docutils literal notranslate"><span class="pre">bootstrap=True</span></code>) and the sampling of the\nfeatures to consider when looking for the best split at each node\n(if <code class="docutils literal notranslate"><span class="pre">max_features</span> <span class="pre">&lt;</span> <span class="pre">n_features</span></code>).\nSee <span class="xref std std-term">Glossary</span> for details.</p>\n</dd>\n<dt>verbose<span class="classifier">int, default=0</span></dt><dd><p>Controls the verbosity when fitting and predicting.</p>\n</dd>\n<dt>warm_start<span class="classifier">bool, default=False</span></dt><dd><p>When set to <code class="docutils literal notranslate"><span class="pre">True</span></code>, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest. See <span class="xref std std-term">Glossary</span> and\n<span class="xref std std-ref">gradient_boosting_warm_start</span> for details.</p>\n</dd>\n<dt>class_weight<span class="classifier">{“balanced”, “balanced_subsample”}, dict or list of dicts,             default=None</span></dt><dd><p>Weights associated with classes in the form <code class="docutils literal notranslate"><span class="pre">{class_label:</span> <span class="pre">weight}</span></code>.\nIf not given, all classes are supposed to have weight one. For\nmulti-output problems, a list of dicts can be provided in the same\norder as the columns of y.</p>\n<p>Note that for multioutput (including multilabel) weights should be\ndefined for each class of every column in its own dict. For example,\nfor four-class multilabel classification weights should be\n[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of\n[{1:1}, {2:5}, {3:1}, {4:1}].</p>\n<p>The “balanced” mode uses the values of y to automatically adjust\nweights inversely proportional to class frequencies in the input data\nas <code class="docutils literal notranslate"><span class="pre">n_samples</span> <span class="pre">/</span> <span class="pre">(n_classes</span> <span class="pre">*</span> <span class="pre">np.bincount(y))</span></code></p>\n<p>The “balanced_subsample” mode is the same as “balanced” except that\nweights are computed based on the bootstrap sample for every tree\ngrown.</p>\n<p>For multi-output, the weights of each column of y will be multiplied.</p>\n<p>Note that these weights will be multiplied with sample_weight (passed\nthrough the fit method) if sample_weight is specified.</p>\n</dd>\n<dt>ccp_alpha<span class="classifier">non-negative float, default=0.0</span></dt><dd><p>Complexity parameter used for Minimal Cost-Complexity Pruning. The\nsubtree with the largest cost complexity that is smaller than\n<code class="docutils literal notranslate"><span class="pre">ccp_alpha</span></code> will be chosen. By default, no pruning is performed. See\n<span class="xref std std-ref">minimal_cost_complexity_pruning</span> for details.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n<dt>max_samples<span class="classifier">int or float, default=None</span></dt><dd><p>If bootstrap is True, the number of samples to draw from X\nto train each base estimator.</p>\n<ul class="simple">\n<li><p>If None (default), then draw <cite>X.shape[0]</cite> samples.</p></li>\n<li><p>If int, then draw <cite>max_samples</cite> samples.</p></li>\n<li><p>If float, then draw <cite>max(round(n_samples * max_samples), 1)</cite> samples. Thus,\n<cite>max_samples</cite> should be in the interval <cite>(0.0, 1.0]</cite>.</p></li>\n</ul>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.22.</span></p>\n</div>\n</dd>\n</dl>\n<dl>\n<dt><a href="#id2"><span class="problematic" id="id3">estimator_</span></a><span class="classifier"><code class="xref py py-class docutils literal notranslate"><span class="pre">DecisionTreeClassifier</span></code></span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.2: </span><cite>base_estimator_</cite> was renamed to <cite>estimator_</cite>.</p>\n</div>\n</dd>\n<dt><a href="#id4"><span class="problematic" id="id5">base_estimator_</span></a><span class="classifier">DecisionTreeClassifier</span></dt><dd><p>The child estimator template used to create the collection of fitted\nsub-estimators.</p>\n<div class="deprecated">\n<p><span class="versionmodified deprecated">Deprecated since version 1.2: </span><cite>base_estimator_</cite> is deprecated and will be removed in 1.4.\nUse <cite>estimator_</cite> instead.</p>\n</div>\n</dd>\n<dt><a href="#id6"><span class="problematic" id="id7">estimators_</span></a><span class="classifier">list of DecisionTreeClassifier</span></dt><dd><p>The collection of fitted sub-estimators.</p>\n</dd>\n<dt><a href="#id8"><span class="problematic" id="id9">classes_</span></a><span class="classifier">ndarray of shape (n_classes,) or a list of such arrays</span></dt><dd><p>The classes labels (single output problem), or a list of arrays of\nclass labels (multi-output problem).</p>\n</dd>\n<dt><a href="#id10"><span class="problematic" id="id11">n_classes_</span></a><span class="classifier">int or list</span></dt><dd><p>The number of classes (single output problem), or a list containing the\nnumber of classes for each output (multi-output problem).</p>\n</dd>\n<dt><a href="#id12"><span class="problematic" id="id13">n_features_in_</span></a><span class="classifier">int</span></dt><dd><p>Number of features seen during <span class="xref std std-term">fit</span>.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 0.24.</span></p>\n</div>\n</dd>\n<dt><a href="#id14"><span class="problematic" id="id15">feature_names_in_</span></a><span class="classifier">ndarray of shape (<cite>n_features_in_</cite>,)</span></dt><dd><p>Names of features seen during <span class="xref std std-term">fit</span>. Defined only when <cite>X</cite>\nhas feature names that are all strings.</p>\n<div class="versionadded">\n<p><span class="versionmodified added">New in version 1.0.</span></p>\n</div>\n</dd>\n<dt><a href="#id16"><span class="problematic" id="id17">n_outputs_</span></a><span class="classifier">int</span></dt><dd><p>The number of outputs when <code class="docutils literal notranslate"><span class="pre">fit</span></code> is performed.</p>\n</dd>\n<dt><a href="#id18"><span class="problematic" id="id19">feature_importances_</span></a><span class="classifier">ndarray of shape (n_features,)</span></dt><dd><p>The impurity-based feature importances.\nThe higher, the more important the feature.\nThe importance of a feature is computed as the (normalized)\ntotal reduction of the criterion brought by that feature.  It is also\nknown as the Gini importance.</p>\n<p>Warning: impurity-based feature importances can be misleading for\nhigh cardinality features (many unique values). See\n<code class="xref py py-func docutils literal notranslate"><span class="pre">sklearn.inspection.permutation_importance()</span></code> as an alternative.</p>\n</dd>\n<dt><a href="#id20"><span class="problematic" id="id21">oob_score_</span></a><span class="classifier">float</span></dt><dd><p>Score of the training dataset obtained using an out-of-bag estimate.\nThis attribute exists only when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n<dt><a href="#id22"><span class="problematic" id="id23">oob_decision_function_</span></a><span class="classifier">ndarray of shape (n_samples, n_classes) or             (n_samples, n_classes, n_outputs)</span></dt><dd><p>Decision function computed with out-of-bag estimate on the training\nset. If n_estimators is small it might be possible that a data point\nwas never left out during the bootstrap. In this case,\n<cite>oob_decision_function_</cite> might contain NaN. This attribute exists\nonly when <code class="docutils literal notranslate"><span class="pre">oob_score</span></code> is True.</p>\n</dd>\n</dl>\n<p>sklearn.tree.DecisionTreeClassifier : A decision tree classifier.\nsklearn.ensemble.ExtraTreesClassifier : Ensemble of extremely randomized</p>\n<blockquote>\n<div><p>tree classifiers.</p>\n</div></blockquote>\n<dl class="simple">\n<dt>sklearn.ensemble.HistGradientBoostingClassifier<span class="classifier">A Histogram-based Gradient</span></dt><dd><p>Boosting Classification Tree, very fast for big datasets (n_samples &gt;=\n10_000).</p>\n</dd>\n</dl>\n<p>The default values for the parameters controlling the size of the trees\n(e.g. <code class="docutils literal notranslate"><span class="pre">max_depth</span></code>, <code class="docutils literal notranslate"><span class="pre">min_samples_leaf</span></code>, etc.) lead to fully grown and\nunpruned trees which can potentially be very large on some data sets. To\nreduce memory consumption, the complexity and size of the trees should be\ncontrolled by setting those parameter values.</p>\n<p>The features are always randomly permuted at each split. Therefore,\nthe best found split may vary, even with the same training data,\n<code class="docutils literal notranslate"><span class="pre">max_features=n_features</span></code> and <code class="docutils literal notranslate"><span class="pre">bootstrap=False</span></code>, if the improvement\nof the criterion is identical for several splits enumerated during the\nsearch of the best split. To obtain a deterministic behaviour during\nfitting, <code class="docutils literal notranslate"><span class="pre">random_state</span></code> has to be fixed.</p>\n<aside class="footnote brackets" id="id1" role="note">\n<span class="label"><span class="fn-bracket">[</span>1<span class="fn-bracket">]</span></span>\n<ol class="upperalpha simple" start="12">\n<li><p>Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.</p></li>\n</ol>\n</aside>\n<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">RandomForestClassifier</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="kn">import</span> <span class="n">make_classification</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">make_classification</span><span class="p">(</span><span class="n">n_samples</span><span class="o">=</span><span class="mi">1000</span><span class="p">,</span> <span class="n">n_features</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>\n<span class="gp">... </span>                           <span class="n">n_informative</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">n_redundant</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span>\n<span class="gp">... </span>                           <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">clf</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">max_depth</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>\n<span class="go">RandomForestClassifier(...)</span>\n<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]]))</span>\n<span class="go">[1]</span>\n</pre></div>\n</div>\n</div></blockquote>\n\n\n</div>',
    outputType: null,
    defaultArgs: {
      n_estimators: 100,
      criterion: 'gini',
    },
  },
];
