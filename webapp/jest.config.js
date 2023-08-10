/**
 * @type {import('@jest/types').Config.InitialOptions}
 */
module.exports = {
  testEnvironment: 'jsdom',
  moduleDirectories: ['node_modules', 'src'],
  transformIgnorePatterns: [
    'node_modules/(?!' + 
      [
        'yaml',
      ].join('|') +
    ')',
  ],
  moduleNameMapper: {
    '@app/(.*)': './src/app/$1',
    '@utils/(.*)': './utils/$1',
    '@utils': 'utils',
    '@model-compiler/(.*)': './src/model-compiler/$1'
  }
};
