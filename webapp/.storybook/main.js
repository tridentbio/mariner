const path = require('path');
const {
  mergeConfig
} = require("vite");
const {
  default: tsconfigPaths
} = require('vite-tsconfig-paths');
module.exports = {
  "stories": ["../src/**/*.stories.mdx", "../src/**/*.stories.@(js|jsx|ts|tsx)"],
  "addons": ["@storybook/addon-links", "@storybook/addon-essentials", "@storybook/addon-interactions", "@storybook/addon-mdx-gfm"],
  "framework": {
    name: "@storybook/react-vite",
    options: {}
  },
  "core": {
    "options": {
      lazyCompilation: true
    }
  },
  viteFinal(config, {
    configType
  }) {
    return mergeConfig(config, {
      plugins: [tsconfigPaths()]
    });
  },
  docs: {
    autodocs: true
  }
};