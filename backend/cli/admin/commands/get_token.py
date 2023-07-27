import click


@click.command()
@click.argument("username", type=str)
@click.argument("password", type=str)
def get_token(username: str, password: str):
    import mariner.users

    auth = mariner.users.BasicAuth(username=username, password=password)
    token = mariner.users.authenticate(basic=auth)
    print(token.json())
