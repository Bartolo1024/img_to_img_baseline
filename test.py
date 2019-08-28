import click

@click.group()
@click.option('-x')
def cli(x):
    print('evaluation')
    pass

@cli.command()
def initdb():
    click.echo('Initialized the database')

@cli.command()
def dropdb():
    click.echo('Dropped the database')

# cli.add_command(initdb)
# cli.add_command(dropdb)

if __name__ == '__main__':
    cli()
