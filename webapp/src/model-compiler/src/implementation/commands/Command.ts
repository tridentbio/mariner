abstract class Command<T, S> {
  args: T

  abstract execute :  () => S

  constructor(args: T) {
    this.args = args
  }


}

export default Command
