from typing import Optional

from risclog.claimxdb.database.base import PortalObject
from sqlalchemy import ARRAY, Column, Date, DateTime, Double, FetchedValue, Index, Integer, Numeric, PrimaryKeyConstraint, String, Table, Text, Time, text
from sqlalchemy.orm import Mapped, mapped_column
import datetime

t_eventlog = Table(
    'eventlog', PortalObject.metadata,
    Column('eventid', Integer),
    Column('eventtyp', String(30)),
    Column('datum', Date),
    Column('uhrzeit', DateTime),
    Column('ukz', String(20)),
    Column('fktname', String(50)),
    Column('man', String(4)),
    Column('request', Text),
    Column('errortype', String(255)),
    Column('errorvalue', Text),
    Column('traceback', Text),
    Column('skriptname', String(50)),
    Column('details', Text),
    Column('scode', Text),
    Index('i_ev_tagesdatum', 'eventtyp', 'datum', postgresql_using=False)
)

class Eventlogadmin(PortalObject):  # type: ignore[misc]
    __tablename__ = 'eventlogadmin'
    __table_args__ = (
        PrimaryKeyConstraint('eventtyp', name='eventlogadmin_pkey'),
        {'comment': 'Admin Event-Log'}
    )

    eventtyp: Mapped[str] = mapped_column(String(30), primary_key=True, comment='Eventtyp')
    ignored_exceptions: Mapped[Optional[str]] = mapped_column(Text, comment='Ignored exceptions, getrennt durch CR')

t_eventmail = Table(
    'eventmail', PortalObject.metadata,
    Column('eventtyp', String(30)),
    Column('skriptname', String(50)),
    Column('fktname', String(50)),
    Column('errortype', String(255))
)

t_importlog = Table(
    'importlog', PortalObject.metadata,
    Column('ukz', String(20), comment='Fuhrparkimport Hochladender Benutzer'),
    Column('man', String(4), comment='Fuhrparkimport angegebener Import-Mandant'),
    Column('importdat', Date, comment='Fuhrparkimport Importdatum'),
    Column('file', String(255), comment='Fuhrparkimport Pfad zur Quelldatei'),
    Column('dl', String(30), comment='Fuhrparkimport angegebener Dienstleister - Import-Schema'),
    Column('typ', String(20), comment='Fuhrparkimport Art Fahrzeuge / Fahrer'),
    Column('info', Text, comment='Fuhrparkimport Fehler/Ereignisse/Hinweise'),
    Column('status', String(30), comment='Fuhrparkimport status - Importiert ?'),
    Column('cnt', Integer, comment='Fuhrparkimport Anzahl importierter Zeilen'),
    Column('importid', Integer),
    Column('zas', Text, comment='Sendungsdatenschnittstelle'),
    Column('abrufdat', DateTime, comment='Datum des Sendungsdatumsabrufes'),
    Column('sndnr', Text, comment='Sendungsdatum'),
    Column('produkt', String(20), comment='Produkt'),
    Column('zasinfo', String(30), comment='Zusatzinfo ZAS'),
    Column('dtyp', String(1), comment='Aktentyp: (S)chaden, (P)rotokoll'),
    Column('snddat', Date),
    Column('abrufsec', Double(53), comment='Abrufdauer in sec'),
    Column('fzgid', String(20)),
    comment='Fehlerlog Fuhrparkimport'
)

class Requestlog(PortalObject):  # type: ignore[misc]
    __tablename__ = 'requestlog'
    __table_args__ = (
        PrimaryKeyConstraint('logid', name='requestlog_pkey'),
        Index('i_logdatum', 'datum', 'zeit', postgresql_using=False),
        Index('i_logman', 'man', postgresql_using=False),
        Index('i_logukz', 'ukz', 'datum', postgresql_using=False)
    )

    logid: Mapped[int] = mapped_column(Integer, primary_key=True)
    ukz: Mapped[str] = mapped_column(String(20))
    man: Mapped[str] = mapped_column(String(4))
    zopeid: Mapped[str] = mapped_column(String(50))
    datum: Mapped[datetime.date] = mapped_column(Date)
    zeit: Mapped[datetime.time] = mapped_column(Time)
    url: Mapped[Optional[str]] = mapped_column(Text)
    query_string: Mapped[Optional[str]] = mapped_column(Text)
    request_method: Mapped[Optional[str]] = mapped_column(String(20))
    http_referer: Mapped[Optional[str]] = mapped_column(Text)
    requestdata: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text()), comment='REQUEST')
    sessiondata: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text()), comment='Daten Usersession')

t_service_temp = Table(
    'service_temp', PortalObject.metadata,
    Column('ukz', String(30)),
    Column('request', Text)
)

t_userliste = Table(
    'userliste', PortalObject.metadata,
    Column('ukz', String(20))
)

__all__ = ['Eventlogadmin', 'Requestlog']
