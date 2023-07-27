import click


@click.command()
@click.argument("username", type=str)
@click.argument("password", type=str)
def get_token(username: str, password: str):
    import mariner.users

    auth = mariner.users.BasicAuth(
        username="admin@mariner.trident.bio", password="123456"
    )
    token = mariner.users.authenticate(basic=auth)
    print(token.json())
