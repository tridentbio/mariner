import { createAsyncThunk, createSlice, current } from '@reduxjs/toolkit';
import { RootState } from '../../app/store';
import { Model } from 'app/types/domain/models';
import {
  EClassPaths,
  ModelOptionsComponent,
  ModelOptions,
} from 'app/types/domain/modelOptions';
import * as expsApi from 'features/models/experimentsApi';
import { UpdateExperiment } from '@app/websocket/handler';
import { modelsApi } from 'app/rtk/models';
import { Experiment, TrainingStage } from 'app/types/domain/experiments';
import { experimentsApi } from 'app/rtk/experiments';
import { enhancedApi } from 'app/rtk/generated/experiments';
import { deepClone } from 'utils';

export type ArgumentType = 'string' | 'bool' | 'int';

export interface ModelState {
  models: Model[];
  totalModels: number;
  experiments: Experiment[];
  modelOptions?: ModelOptions[];
  progress?: number;
}

export const startTraning = createAsyncThunk(
  'models/startTraning',
  expsApi.startTraning
);

export const fetchExperiments = createAsyncThunk(
  'models/getExperiments',
  expsApi.fetchExperiments
);

export const fetchExperimentsProgress = createAsyncThunk(
  'models/getExperimentsProgress',
  expsApi.fetchRunningExperimentsHistory
);

const initialState: ModelState = {
  models: [],
  experiments: [],
  totalModels: 0,
};

export const modelSlice = createSlice({
  name: 'model',
  initialState,
  reducers: {
    addTraining: (
      state,
      action: {
        type: string;
        payload: Experiment;
      }
    ) => {
      if (action.payload) {
        state.experiments.unshift(action.payload);
      }
    },
    updateExperiment: (
      state,
      action: { type: string; payload: UpdateExperiment['data'] }
    ) => {
      state.experiments.forEach((exp) => {
        if (exp.id === action.payload.experimentId) {
          if (!exp.trainMetrics) exp.trainMetrics = {};
          if (action.payload.metrics) {
            exp.trainMetrics = { ...action.payload.metrics };
            exp.valMetrics = { ...action.payload.metrics };
          }
          if (exp.epochs)
            exp.progress = (action.payload.epoch || 0) / exp.epochs;
          if (action.payload.stage) {
            if (action.payload.stage !== 'RUNNING') exp.progress = 1;
            exp.stage = action.payload.stage as TrainingStage;
          }
        }
      });
    },
    updateExperiments: (
      state,
      action: { type: string; payload: Experiment[] }
    ) => {
      state.experiments = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(fetchExperiments.fulfilled, (state, action) => {
      state.experiments = action.payload.data.map((exp) => {
        if (exp.stage !== 'RUNNING') {
          return { ...exp, progress: 1.0 };
        }
        return exp;
      });
    });

    builder.addCase(fetchExperimentsProgress.fulfilled, (state, action) => {
      const experiments = current(state).experiments;
      const updatedExperiments = deepClone(experiments) as typeof experiments;
      for (const experimentProgress of action.payload) {
        let foundIndex: number | undefined = undefined;
        const foundExperiment = updatedExperiments.find((exp, index) => {
          if (exp.id === experimentProgress.experimentId) {
            foundIndex = index;
            return true;
          }
          return false;
        });
        if (foundExperiment) {
          const epoch = experimentProgress.runningHistory['train_loss']?.length;
          if (foundExperiment.epochs && typeof foundIndex === 'number') {
            updatedExperiments[foundIndex as number].progress =
              epoch / foundExperiment.epochs;
            state.experiments = updatedExperiments;
          }
        }
      }
    });

    builder.addCase(startTraning.fulfilled, (state, action) => {
      state.experiments = [...state.experiments, action.payload];
    });

    builder.addMatcher(
      modelsApi.endpoints.getModelsOld.matchFulfilled,
      (state, action) => {
        state.models = action.payload.data;
        state.totalModels = action.payload.total;
      }
    );

    builder.addMatcher(
      modelsApi.endpoints.getModelById.matchFulfilled,
      (state, action) => {
        if (state.models.find((model) => model.id === action.payload.id))
          return;
        state.models.push(action.payload);
      }
    );
    builder.addMatcher(
      modelsApi.endpoints.createModelOld.matchFulfilled,
      (state, action) => {
        state.models.push(action.payload);
        state.totalModels += 1;
      }
    );

    builder.addMatcher(
      modelsApi.endpoints.getOptions.matchFulfilled,
      (state, action) => {
        state.modelOptions = action.payload;
      }
    );
    builder.addMatcher(
      experimentsApi.endpoints.getExperiments.matchFulfilled,
      (state, action) => {
        state.experiments = action.payload.data;
      }
    );

    builder.addMatcher(
      enhancedApi.endpoints.getExperiments.matchFulfilled,
      (state, action) => {
        // @ts-ignore
        state.experiments = action.payload.data;
      }
    );

    builder.addMatcher(
      enhancedApi.endpoints.postExperiments.matchFulfilled,
      (state, action) => {
        // @ts-ignore
        state.experiments = [...state.experiments, action.payload];
      }
    );
  },
});

export const selectModelByName =
  (modelName: string | undefined) => (state: RootState) =>
    state.models.models.find((model) => model.name === modelName);
export const selectModelById =
  (modelId: number | undefined) => (state: RootState) =>
    state.models.models.find((model) => model.id === modelId);

export type ComponentArgsTypeDict = {
  [key in EClassPaths]: ModelOptionsComponent['constructorArgsSummary'];
};

export const splitModelOptionsInLayersFeaturizers = (
  modelOptions: ModelOptions[]
) => {
  return modelOptions.reduce(
    (acc, model) => {
      if (model.type === 'layer') acc.layers.push(model);
      if (model.type === 'featurizer') acc.featurizers.push(model);
      return acc;
    },
    { layers: [], featurizers: [] } as {
      layers: ModelOptions[];
      featurizers: ModelOptions[];
    }
  );
};

export const selectModel = (modelName: string) => (state: RootState) => {
  const model = state.models.models.find((model) => model.name === modelName);

  return model;
};

export const selectExperiments =
  (query: { modelId: number }) => (state: RootState) => {
    const experiments = state.models.experiments.filter(
      (exp) => exp.modelVersion.modelId === query.modelId
    );
    return [...experiments].reverse();
  };

export const { updateExperiment, addTraining, updateExperiments } =
  modelSlice.actions;
export default modelSlice.reducer;
