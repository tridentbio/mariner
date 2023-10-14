abstract class Command<T, S> {
  args: T;

  abstract execute: (schema: S) => S;

  constructor(args: T) {
    this.args = args;
  }
}

export default Command;
