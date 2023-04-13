module.exports = {
  testEnvironment: 'jsdom',
  moduleDirectories: ['node_modules', 'src'],
  moduleNameMapper: {
    '@app/(.*)': './src/app/$1',
    '@utils/(.*)': './utils/$1',
    '@utils': 'utils'
  }
};
